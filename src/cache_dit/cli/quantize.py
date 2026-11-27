"""`--quantize-params` / `--svdq-params` resolution and quantization application."""

from typing import Any, Dict, List, Optional

import torch
from diffusers import DiffusionPipeline
from diffusers.quantizers import PipelineQuantizationConfig

from ..caching import BlockAdapter
from ..logger import init_logger
from ..quantization import quantize, QuantizeConfig
from ..quantization.backend import QuantizeBackend
from ..quantization.config import DEFAULT_EXCLUDE_LAYERS
from ..quantization.config import _resolve_svdq_kwargs
from ..quantization.config import TEXT_ENCODER_QUANT_TYPES
from ..utils import assign_transformer
from ..utils import declare_repeated_blocks
from ..utils import iter_transformer_modules
from ..utils import resolve_transformer
from .params import (
  parse_param_pairs,
  resolve_param_pairs,
  QUANTIZE_PARAM_SPECS,
  SVDQ_PARAM_SPECS,
)
from .parallel import get_rank_device, get_text_encoder_from_pipe

logger = init_logger(__name__)


def parse_quant_communicate_params(value: str) -> Optional[Dict[str, str]]:
  """Parse `--quant-communicate-params "k=v,k=v"` into raw options.

  :param value: Raw argument value.
  :returns: Raw string options, or None for an empty value.
  """
  return parse_param_pairs(value, "--quant-communicate-params", "'bits=6,quant_type=per_group_32'")


def _base_svdq_kwargs(args) -> Dict[str, Any]:
  """SVDQ kwargs inherited from the generic `--layerwise-*` flags and the CLI defaults."""

  return {
    "smooth_strategy": "identity",
    "calibrate_precision": "low",
    "runtime_kernel": None,
    "lora_layout": None,
    "quantize_device": "cuda",
    "offload_quantized_layers_to_cpu": False,
    "layerwise_offload": False,
    "async_transfer": args.layerwise_async_transfer,
    "transfer_buckets": args.layerwise_transfer_buckets,
    "prefetch_limit": args.layerwise_prefetch_limit,
    "max_copy_streams": args.layerwise_max_copy_streams,
    "max_inflight_prefetch_bytes": args.layerwise_max_inflight_prefetch_bytes,
    "persistent_buckets": args.layerwise_persistent_buckets,
    "persistent_bins": args.layerwise_persistent_bins,
    "defer_move_to_execution_device": False,
    "few_shot_steps": 1,
    "few_shot_relax_factor": 1.5,
    "few_shot_relax_top_ratio": 0.25,
    "few_shot_relax_strategy": "auto",
    "few_shot_auto_compile": False,
    "fused_mlp": False,
  }


def _derive_cli_svdq_kwargs(kwargs: Dict[str, Any], args, explicit: set) -> None:
  """Apply the CLI rules that tie SVDQ options to each other and to `--compile`.

  A key the user passed explicitly always wins over the derived value.

  :param kwargs: Resolved SVDQ kwargs, updated in place.
  :param args: Parsed CLI namespace.
  :param explicit: Keys the user passed on the command line.
  """

  if kwargs["layerwise_offload"]:
    kwargs["offload_quantized_layers_to_cpu"] = True
  few_shot = kwargs["smooth_strategy"] == "few_shot"
  if "few_shot_auto_compile" not in explicit and few_shot and args.compile:
    kwargs["few_shot_auto_compile"] = True
  if ("defer_move_to_execution_device" not in explicit and few_shot
      and kwargs["offload_quantized_layers_to_cpu"]):
    kwargs["defer_move_to_execution_device"] = True


def _resolve_cli_quantization(args) -> None:
  """Resolve `--quantize-params` / `--svdq-params` into `args`.

  Any `--quantize-params` key other than `text_encoder` turns on transformer quantization, so
  `args.quantize` is derived here. `args.quantize_fields` and `args.svdq_kwargs` hold the
  resolved options and are what the quantization helpers read; `quantize_type` and
  `quantize_exclude_layers` are also exposed as plain attributes because examples set them
  programmatically.
  """

  fields = resolve_param_pairs(args.quantize_params, QUANTIZE_PARAM_SPECS, "--quantize-params")

  # `text_encoder` on its own only quantizes the text encoder.
  args.quantize = bool(set(fields) - {"text_encoder"})
  args.quantize_type = (fields.get("quant_type") or "float8_per_row") if args.quantize else None
  if fields.get("text_encoder") is True:
    if args.quantize_type is None:
      raise ValueError("--quantize-params 'text_encoder=true' needs a transformer quant_type, "
                       "e.g. 'quant_type=float8_per_row,text_encoder=true'.")
    if args.quantize_type not in TEXT_ENCODER_QUANT_TYPES:
      raise ValueError(
        f"--quantize-params 'text_encoder=true' got quant_type {args.quantize_type}, "
        f"which is not a text encoder type: {TEXT_ENCODER_QUANT_TYPES}.")
    fields["text_encoder"] = args.quantize_type

  args.quantize_exclude_layers = fields.get("exclude_layers")
  args.quantize_fields = fields

  is_svdq = args.quantize_type is not None and args.quantize_type.startswith("svdq")
  raw_svdq = args.svdq_params or {}
  if raw_svdq and not is_svdq:
    raise ValueError(f"--svdq-params requires an svdq_* quant_type, got {args.quantize_type!r}.")
  kwargs = None
  if is_svdq:
    kwargs = _base_svdq_kwargs(args)
    kwargs.update(resolve_param_pairs(raw_svdq, SVDQ_PARAM_SPECS, "--svdq-params"))
    _derive_cli_svdq_kwargs(kwargs, args, set(raw_svdq))
    # Fail on a bad value here rather than after the model has been loaded.
    _resolve_svdq_kwargs(kwargs)
  args.svdq_kwargs = kwargs


def _should_defer_few_shot_pipeline_move(transformer) -> bool:
  if transformer is None or not getattr(transformer, "_svdq_pending_quantization", False):
    return False
  controller = getattr(transformer, "_svdq_few_shot_controller", None)
  if controller is None:
    return False
  if controller.context.svdq_kwargs.get("smooth_strategy") != "few_shot":
    return False
  if not bool(controller.context.svdq_kwargs.get("defer_move_to_execution_device", False)):
    return False
  return bool(
    controller.context.svdq_kwargs.get("layerwise_offload", False)
    or controller.context.svdq_kwargs.get("offload_quantized_layers_to_cpu", False))


def _should_skip_initial_few_shot_pipeline_move(transformer) -> bool:
  if transformer is None or not getattr(transformer, "_svdq_pending_quantization", False):
    return False
  controller = getattr(transformer, "_svdq_few_shot_controller", None)
  if controller is None:
    return False
  if controller.context.svdq_kwargs.get("smooth_strategy") != "few_shot":
    return False
  return bool(controller.context.svdq_kwargs.get("layerwise_offload", False))


def _register_deferred_few_shot_pipeline_move_callbacks(args, pipe) -> bool:
  if (pipe is None or args.device_map_balance or args.cpu_offload or args.sequential_cpu_offload
      or args.module_layerwise_cpu_offload):
    return False

  tracked_names: list[str] = []
  for name in ("transformer", "transformer_2"):
    transformer = getattr(pipe, name, None)
    if _should_defer_few_shot_pipeline_move(transformer):
      tracked_names.append(name)

  if not tracked_names:
    return False

  _, device = get_rank_device()
  target_device = torch.device(device)

  def _register_for_transformer(name: str) -> None:
    transformer = getattr(pipe, name, None)
    if transformer is None:
      return

    callbacks = getattr(transformer, "_svdq_post_quantize_callbacks", None)
    if callbacks is None:
      transformer._svdq_post_quantize_callbacks = []
      callbacks = transformer._svdq_post_quantize_callbacks
    callback_names = getattr(transformer, "_svdq_post_quantize_callback_names", None)
    if callback_names is None:
      transformer._svdq_post_quantize_callback_names = set()
      callback_names = transformer._svdq_post_quantize_callback_names
    callback_name = f"pipeline_move:{name}:{target_device}"
    if callback_name in callback_names:
      return

    def _callback(_quantized_module):
      remaining = [
        tracked_name for tracked_name in tracked_names
        if getattr(getattr(pipe, tracked_name, None), "_svdq_pending_quantization", False)
      ]
      if remaining:
        logger.info(
          "SVDQ few-shot quantization finished for %s; waiting for remaining modules before the "
          "final pipeline move: %s.",
          name,
          ", ".join(remaining),
        )
        return
      current_transformer = getattr(pipe, name, None)
      runtime_layerwise_handle = getattr(
        current_transformer,
        "_svdq_runtime_layerwise_offload_handle",
        None,
      )
      if runtime_layerwise_handle is not None:
        pipe._svdq_move_to_device_after_forward = target_device
        logger.info(
          "SVDQ few-shot quantization finished for %s; keeping temporary runtime layerwise "
          "offload active until the current pipeline call returns, then moving the pipeline to "
          "%s.",
          name,
          target_device,
        )
        return
      try:
        logger.info(
          "SVDQ few-shot quantization finished; moving pipeline to %s immediately so the "
          "current pipeline call can continue on the execution device.",
          target_device,
        )
        pipe.to(target_device)
      except Exception as exc:
        pipe._svdq_move_to_device_after_forward = target_device
        logger.warning(
          "Immediate pipeline move to %s after SVDQ few-shot quantization failed with %s: %s. "
          "Falling back to a deferred move after the current pipeline call returns.",
          target_device,
          type(exc).__name__,
          exc,
        )

    callback_names.add(callback_name)
    callbacks.append(_callback)

  for name in tracked_names:
    _register_for_transformer(name)

  logger.info(
    "Deferring final pipeline move to %s until SVDQ few-shot quantization finishes for %s.",
    target_device,
    ", ".join(tracked_names),
  )
  return True


def maybe_finalize_deferred_svdq_pipe_move(
  pipe_or_adapter: DiffusionPipeline | BlockAdapter, ) -> bool:
  if isinstance(pipe_or_adapter, BlockAdapter):
    pipe = pipe_or_adapter.pipe
  else:
    pipe = pipe_or_adapter
  if pipe is None:
    return False

  target_device = getattr(pipe, "_svdq_move_to_device_after_forward", None)
  if target_device is None:
    return False

  delattr(pipe, "_svdq_move_to_device_after_forward")
  for name in ("transformer", "transformer_2"):
    transformer = getattr(pipe, name, None)
    handle = getattr(transformer, "_svdq_runtime_layerwise_offload_handle", None)
    if handle is None:
      continue
    logger.info(
      "Removing temporary runtime layerwise offload from %s before moving pipeline to %s.",
      name,
      target_device,
    )
    handle.remove()
    delattr(transformer, "_svdq_runtime_layerwise_offload_handle")
  logger.info(
    "Moving pipeline to %s after SVDQ few-shot runtime quantization completed.",
    target_device,
  )
  pipe.to(target_device)
  return True


def _resolve_cli_quantize_exclude_layers(args) -> Optional[List[str]]:
  """Merge `--quantize-params exclude_layers` with the default exclusion table.

  The flag adds patterns instead of replacing the table: the defaults guard embeddings and
  modulation tables, which cost far more accuracy than the memory they save, and a model-specific
  list (MiniMax-H3 keeps its float32 projections and per-token AdaLN tables out) must not silently
  drop them.

  :param args: CLI arguments.
  :returns: The merged patterns, or None when the flag is absent.
  """
  extra = getattr(args, "quantize_exclude_layers", None)
  if not extra:
    return None
  merged = list(DEFAULT_EXCLUDE_LAYERS)
  merged.extend(name for name in extra if name not in merged)
  return merged


def _quantize_fields(args) -> Dict[str, Any]:
  """Return the resolved `--quantize-params`, or an empty mapping when post-processing was skipped.

  :param args: Parsed CLI namespace.
  :returns: The resolved options.
  """

  return getattr(args, "quantize_fields", None) or {}


def _resolve_cli_quantize_backend(args) -> QuantizeBackend:
  """Resolve the quantization backend, mapping an omitted value to AUTO.

  :param args: Parsed CLI namespace.
  :returns: Resolved backend enum.
  """

  value = _quantize_fields(args).get("backend")
  return QuantizeBackend.AUTO if value is None else QuantizeBackend.from_str(value)


def build_quantize_config(args, **overrides) -> QuantizeConfig:
  """Build the transformer `QuantizeConfig` from the resolved CLI quantization options.

  :param args: Parsed CLI namespace, post-processed or with `quantize_type` set directly.
  :param overrides: Fields that replace the CLI values.
  :returns: The validated config.
  """

  svdq_kwargs = getattr(args, "svdq_kwargs", None)
  # `quantize_type` is None when only the text encoder is quantized (`text_encoder` override below).
  if (args.quantize_type or "").startswith("svdq") and svdq_kwargs is None:
    # `quantize_type` was set after post-processing, so no SVDQ options were resolved yet.
    svdq_kwargs = _base_svdq_kwargs(args)
    _derive_cli_svdq_kwargs(svdq_kwargs, args, explicit=set())
  fields = _quantize_fields(args)
  return QuantizeConfig(
    **{
      "backend": _resolve_cli_quantize_backend(args),
      "quant_type": args.quantize_type,
      "regional_quantize": fields.get("regional_quantize", True),
      "per_tensor_fallback": fields.get("per_tensor_fallback", True),
      "exclude_layers": _resolve_cli_quantize_exclude_layers(args),
      "svdq_kwargs": svdq_kwargs,
      "fp8_fused_mlp": bool(fields.get("fp8_fused_mlp")),
      "fp4_fused_mlp": bool(fields.get("fp4_fused_mlp")),
      "mxfp4_quantize_device": fields.get("mxfp4_quantize_device", "auto"),
      "verbose": bool(fields.get("verbose")),
      **overrides,
    })


def maybe_quantize_transformer(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  # Quantize transformer by default if quantization is enabled
  if args.quantize:
    backend = _resolve_cli_quantize_backend(args)
    if (args.compile and args.cuda_graph and args.quantize_type == "float8_per_row"
        and backend == QuantizeBackend.TORCHAO):
      from ..quantization.torchao.scaled_mm import (
        enable_opaque_torchao_float8_scaled_mm, )

      enable_opaque_torchao_float8_scaled_mm()

    if args.quantize_type in ("bitsandbytes_4bit", "bnb_4bit"):
      logger.debug("bitsandbytes_4bit quantization should be handled by"
                   " PipelineQuantizationConfig in from_pretrained.")
      return pipe_or_adapter

    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please quantize transformer manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    # A pipeline may keep its denoising transformer under another name — MiniMax-H3 loads
    # `transformer_ref` for `ref2va` — and only ever one of the two exists.
    transformer_name, transformer = resolve_transformer(pipe)
    # Regional scoping needs `_repeated_blocks`, which a few diffusers models never declare.
    declare_repeated_blocks(transformer)
    if transformer is None:
      logger.warning("quantize is set but no transformer found in the pipeline.")
    elif not isinstance(transformer, torch.nn.Module):
      logger.warning("Cannot quantize %s: %s is not a torch.nn.Module.", transformer_name,
                     transformer.__class__.__name__)
    else:
      logger.info("Quantizing transformer module '%s' (%s) to %s (backend=%s) ...",
                  transformer_name, transformer.__class__.__name__, args.quantize_type,
                  backend.name)
      transformer = quantize(
        transformer,
        quantize_config=build_quantize_config(args),
      )
      assign_transformer(pipe, transformer, name=transformer_name)

    for secondary_name, secondary in iter_transformer_modules(pipe)[1:]:
      declare_repeated_blocks(secondary)
      logger.info("Quantizing transformer module '%s' (%s) to %s (backend=%s) ...", secondary_name,
                  secondary.__class__.__name__, args.quantize_type, backend.name)
      secondary = quantize(
        secondary,
        quantize_config=build_quantize_config(args),
      )
      assign_transformer(pipe, secondary, name=secondary_name)

  return pipe_or_adapter


def maybe_quantize_text_encoder(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  # Quantize text encoder by default if quantize_text_encoder is enabled
  text_encoder_type = _quantize_fields(args).get("text_encoder")
  if text_encoder_type is not None:
    if text_encoder_type in ("bitsandbytes_4bit", "bnb_4bit"):
      logger.debug("bitsandbytes_4bit quantization should be handled by"
                   " PipelineQuantizationConfig in from_pretrained.")
      return pipe_or_adapter

    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please quantize text encoder manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    text_encoder, name = get_text_encoder_from_pipe(pipe)
    if text_encoder is not None:
      text_encoder_cls_name = text_encoder.__class__.__name__
      if isinstance(text_encoder, torch.nn.Module):
        logger.info(f"Quantizing text encoder module: {name}:{text_encoder_cls_name} to"
                    f" {text_encoder_type} ...")
        text_encoder = quantize(
          text_encoder,
          quantize_config=build_quantize_config(
            args,
            quant_type=text_encoder_type,
            svdq_kwargs=None,
            fp8_fused_mlp=False,
            fp4_fused_mlp=False,
            exclude_layers=list(DEFAULT_EXCLUDE_LAYERS),
          ),
        )
        setattr(pipe, name, text_encoder)
      else:
        logger.warning(f"Cannot quantize text encoder module: {name}:{text_encoder_cls_name} Not a"
                       " torch.nn.Module.")
    else:
      logger.warning("quantize is set but no text encoder found in the pipeline.")
  return pipe_or_adapter


def pipe_quant_bnb_4bit_config(
  args,
  components_to_quantize: Optional[List[str]] = ["text_encoder"],
) -> Optional[PipelineQuantizationConfig]:
  text_encoder_type = _quantize_fields(args).get("text_encoder")
  if text_encoder_type is None and not args.quantize:
    return None

  if components_to_quantize:
    # Remove all components if quantize type is not bitsandbytes_4bit
    if args.quantize_type != "bitsandbytes_4bit":
      if "transformer" in components_to_quantize:
        components_to_quantize.remove("transformer")
      if "transformer_2" in components_to_quantize:
        components_to_quantize.remove("transformer_2")
    if text_encoder_type != "bitsandbytes_4bit":
      if "text_encoder" in components_to_quantize:
        components_to_quantize.remove("text_encoder")
      if "text_encoder_2" in components_to_quantize:
        components_to_quantize.remove("text_encoder_2")

    # Remove text encoder if parallel_text_encoder is enabled
    if args.parallel_text_encoder:
      if "text_encoder" in components_to_quantize:
        components_to_quantize.remove("text_encoder")
      if "text_encoder_2" in components_to_quantize:
        components_to_quantize.remove("text_encoder_2")

  if components_to_quantize:
    quantization_config = ((PipelineQuantizationConfig(
      quant_backend="bitsandbytes_4bit",
      quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
      },
      components_to_quantize=components_to_quantize,
    )) if args.quantize or text_encoder_type is not None else None)
  else:
    quantization_config = None

  return quantization_config
