"""torch.compile application for the transformer, text encoder, ControlNet and VAE."""

from typing import Any, Dict, Optional

import torch
from diffusers import DiffusionPipeline

from ..caching import BlockAdapter
from ..compile.utils import (
  set_compile_configs,
  mindiesd_compile_available,
  mindiesd_compile,
)
from ..logger import init_logger
from ..patches import maybe_patch_nucleus_moe_grouped_mm_offsets
from ..utils import declare_repeated_blocks
from ..utils import iter_transformer_modules
from .parallel import get_text_encoder_from_pipe

logger = init_logger(__name__)


def _compile_mode(args):
  if args.max_autotune:
    if args.cuda_graph:
      return "max-autotune"
    return "max-autotune-no-cudagraphs"
  if args.force_compile_dynamic:
    return None  # let torch.compile to decide itself.
  return None


def _compile_options(args) -> Optional[Dict[str, Any]]:
  if args.cuda_graph and not args.max_autotune:
    return {"triton.cudagraphs": True}
  return None


def _force_compile_dynamic(args, pipe) -> bool:
  _class_maybe_force_compile_dynamic = [
    "QwenImage",
    # `_QwenImage21Pipeline` is the cache-dit optimisation pass over `QwenImage21Pipeline`
    # and must compile with the same `dynamic` decision as the stock pipeline.
    "_QwenImage21",
    # Same rule for `_Flux2KleinPipeline` against `Flux2KleinPipeline`.
    "_Flux2Klein",
    "Flux2KleinKV",
  ]
  if args.force_compile_dynamic:
    return True
  if pipe.__class__.__name__.startswith("CausVid"):
    # A causal autoregressive block is called with a chunk-varying `current_start` token offset, so
    # a dynamic graph is what lets one compiled graph cover all 7 chunks (28 forwards) instead of
    # specialising one per chunk.
    if (getattr(args, "module_layerwise_cpu_offload", False)
        or getattr(args, "sequential_cpu_offload", False)):
      # Offload swaps parameter storages between forwards; a dynamic block graph keeps references
      # that the swap invalidates (measured: `cudaErrorIllegalAddress`). Stay static and accept the
      # per-chunk specialisation, which is still faster than eager here.
      return False
    return True
  if args.parallel_type is None or "tp" not in args.parallel_type.lower():
    return False
  # For some specific pipelines with PyTorch native TP.
  # Auto check if the pipeline class name starts with any
  # of the specified prefixes to decide whether to force
  # compile dynamic.
  return any(
    [pipe.__class__.__name__.startswith(prefix) for prefix in _class_maybe_force_compile_dynamic])


def _compile_transformer_module(args, pipe, transformer, name):
  # Before any early return: a model that is already compiled cannot take the patch anymore, and the
  # warning is the only hint a user gets for why the compile failed.
  maybe_patch_nucleus_moe_grouped_mm_offsets(transformer)

  if transformer is None or isinstance(
      transformer,
      torch._dynamo.OptimizedModule,  # already compiled
  ):
    logger.warning(f"{name} module is already compiled or None, skipping compilation.")
    return transformer

  from diffusers import ModelMixin

  transformer_cls_name = transformer.__class__.__name__
  if not isinstance(transformer, (torch.nn.Module, ModelMixin)):
    logger.warning(f"Cannot compile {name} module: {transformer_cls_name} Not a torch.nn.Module.")
    return transformer

  # Auto-enable MindieSDBackend on NPU
  if mindiesd_compile_available():
    compiled = mindiesd_compile(transformer, name, transformer_cls_name)
    if compiled is not None:
      setattr(pipe, name, compiled)
      return compiled

  # `compile_repeated_blocks()` raises when `_repeated_blocks` is empty, which is the default
  # for diffusers transformers that never declare their denoising stack (e.g. PRX).
  use_regional_compile = (not args.disable_compile_repeated_blocks
                          and hasattr(transformer, "compile_repeated_blocks")
                          and bool(getattr(transformer, "_repeated_blocks", None)))

  if args.cuda_graph and use_regional_compile:
    use_regional_compile = False

  if use_regional_compile:
    logger.info(f"Compiling repeated blocks in {name}: {transformer_cls_name} ...")
    transformer.compile_repeated_blocks(
      fullgraph=args.compile_full_graph,
      mode=_compile_mode(args),
      dynamic=_force_compile_dynamic(args, pipe),
      options=_compile_options(args),
    )
    setattr(pipe, name, transformer)
    return transformer

  if args.cuda_graph:
    logger.info(f"Compiling full {name}: {transformer_cls_name} with CUDA Graph enabled ...")
  else:
    logger.info(f"Compiling full {name}: {transformer_cls_name} ...")
  transformer = torch.compile(
    transformer,
    fullgraph=args.compile_full_graph,
    mode=_compile_mode(args),
    dynamic=_force_compile_dynamic(args, pipe),
    options=_compile_options(args),
  )
  setattr(pipe, name, transformer)
  return transformer


def _register_deferred_few_shot_compile_callback(args, pipe, transformer, name) -> None:
  controller = getattr(transformer, "_svdq_few_shot_controller", None)
  if controller is None:
    return

  def _register_named_callback(callback, callback_name: str) -> bool:
    callbacks = getattr(transformer, "_svdq_post_quantize_callbacks", None)
    if callbacks is None:
      transformer._svdq_post_quantize_callbacks = []
      callbacks = transformer._svdq_post_quantize_callbacks
    callback_names = getattr(transformer, "_svdq_post_quantize_callback_names", None)
    if callback_names is None:
      transformer._svdq_post_quantize_callback_names = set()
      callback_names = transformer._svdq_post_quantize_callback_names
    if callback_name in callback_names:
      return False
    callback_names.add(callback_name)
    callbacks.append(callback)
    return True

  def _callback(_quantized_module):
    current_transformer = getattr(pipe, name, None)
    if current_transformer is None:
      logger.warning("Deferred few-shot compile skipped because %s is no longer present.", name)
      return
    set_compile_configs(cuda_graphs=args.cuda_graph, ulysses_anything=args.ulysses_anything)
    torch.set_float32_matmul_precision("high")
    _compile_transformer_module(args, pipe, current_transformer, name)

  callback_name = f"compile:{name}"
  if not _register_named_callback(_callback, callback_name):
    return
  logger.info("Deferring compile for %s until SVDQ few-shot quantization finishes.", name)


def _should_defer_few_shot_compile(args, transformer) -> bool:
  if transformer is None or not getattr(transformer, "_svdq_pending_quantization", False):
    return False
  controller = getattr(transformer, "_svdq_few_shot_controller", None)
  if controller is None:
    return False
  smooth_strategy = controller.context.svdq_kwargs.get("smooth_strategy")
  if smooth_strategy != "few_shot":
    return False
  return bool(controller.context.svdq_kwargs.get("few_shot_auto_compile", False))


def maybe_compile_transformer(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  few_shot_auto_compile = bool((getattr(args, "svdq_kwargs", None)
                                or {}).get("few_shot_auto_compile"))
  if args.compile or few_shot_auto_compile:
    set_compile_configs(cuda_graphs=args.cuda_graph, ulysses_anything=args.ulysses_anything)
    torch.set_float32_matmul_precision("high")

    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please compile transformer manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    transformers = iter_transformer_modules(pipe)
    if not transformers:
      logger.warning("compile is set but no transformer found in the pipeline.")
    for transformer_name, transformer in transformers:
      declare_repeated_blocks(transformer)
      if _should_defer_few_shot_compile(args, transformer):
        _register_deferred_few_shot_compile_callback(args, pipe, transformer, transformer_name)
      else:
        _compile_transformer_module(args, pipe, transformer, transformer_name)

  return pipe_or_adapter


def maybe_compile_text_encoder(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  if args.compile_text_encoder:
    torch.set_float32_matmul_precision("high")

    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please compile text encoder manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    text_encoder, name = get_text_encoder_from_pipe(pipe)
    if text_encoder is not None and not isinstance(
        text_encoder,
        torch._dynamo.OptimizedModule,  # already compiled
    ):
      # Find module to be compiled, [encoder, model, model.language_model, ...]
      _module_to_compile = text_encoder
      if hasattr(_module_to_compile, "model"):
        if hasattr(_module_to_compile.model, "language_model"):
          _module_to_compile = _module_to_compile.model.language_model
        else:
          _module_to_compile = _module_to_compile.model

      if hasattr(_module_to_compile, "encoder"):
        _module_to_compile = _module_to_compile.encoder

      _module_to_compile_cls_name = _module_to_compile.__class__.__name__
      _text_encoder_cls_name = text_encoder.__class__.__name__
      if isinstance(_module_to_compile, torch.nn.Module):
        logger.info(f"Compiling text encoder module {name}:{_text_encoder_cls_name}:"
                    f"{_module_to_compile_cls_name} ...")
        _module_to_compile = torch.compile(
          _module_to_compile,
          mode=_compile_mode(args),
          options=_compile_options(args),
        )
        # Set back the compiled text encoder
        if hasattr(text_encoder, "model"):
          if hasattr(text_encoder.model, "language_model"):
            text_encoder.model.language_model = _module_to_compile
          else:
            text_encoder.model = _module_to_compile
        if hasattr(text_encoder, "encoder"):
          text_encoder.encoder = _module_to_compile

        setattr(pipe, name, text_encoder)
      else:
        logger.warning(f"Cannot compile text encoder module {name}:{_text_encoder_cls_name}:"
                       f"{_module_to_compile_cls_name} Not a torch.nn.Module.")
    else:
      logger.warning("compile-text-encoder is set but no text encoder found in the pipeline.")
  return pipe_or_adapter


def maybe_compile_controlnet(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  if args.compile_controlnet:
    torch.set_float32_matmul_precision("high")

    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please compile transformer manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    if hasattr(pipe, "controlnet"):
      controlnet = getattr(pipe, "controlnet", None)
      if controlnet is not None and not isinstance(
          controlnet,
          torch._dynamo.OptimizedModule,  # already compiled
      ):
        controlnet_cls_name = controlnet.__class__.__name__
        if isinstance(controlnet, torch.nn.Module):
          logger.info(f"Compiling controlnet module: {controlnet_cls_name} ...")
          controlnet = torch.compile(
            controlnet,
            mode=_compile_mode(args),
            options=_compile_options(args),
          )
          setattr(pipe, "controlnet", controlnet)
        else:
          logger.warning(f"Cannot compile controlnet module: {controlnet_cls_name} Not a"
                         " torch.nn.Module.")
      setattr(pipe, "controlnet", controlnet)
    else:
      logger.warning("compile is set but no controlnet found in the pipeline.")


def maybe_compile_vae(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  if args.compile_vae:
    torch.set_float32_matmul_precision("high")

    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please compile VAE manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    if hasattr(pipe, "vae"):
      vae = getattr(pipe, "vae", None)
      if vae is not None and not isinstance(
          vae,
          torch._dynamo.OptimizedModule,  # already compiled
      ):
        vae_cls_name = vae.__class__.__name__
        if hasattr(vae, "encoder"):
          _encoder_to_compile = vae.encoder
          if isinstance(_encoder_to_compile, torch.nn.Module):
            logger.info(f"Compiling VAE encoder module: {vae_cls_name}.encoder ...")
            vae.encoder = torch.compile(
              _encoder_to_compile,
              mode=_compile_mode(args),
              options=_compile_options(args),
            )
          else:
            logger.warning(f"Cannot compile VAE encoder module: {vae_cls_name}.encoder Not a"
                           " torch.nn.Module.")
        if hasattr(vae, "decoder"):
          _decoder_to_compile = vae.decoder
          if isinstance(_decoder_to_compile, torch.nn.Module):
            logger.info(f"Compiling VAE decoder module: {vae_cls_name}.decoder ...")
            vae.decoder = torch.compile(
              _decoder_to_compile,
              mode=_compile_mode(args),
              options=_compile_options(args),
            )
          else:
            logger.warning(f"Cannot compile VAE decoder module: {vae_cls_name}.decoder Not a"
                           " torch.nn.Module.")
        setattr(pipe, "vae", vae)
      else:
        logger.warning(f"Cannot compile VAE module: {vae_cls_name} Not a torch.nn.Module.")
    else:
      logger.warning("compile-vae is set but no VAE found in the pipeline.")
  return pipe_or_adapter
