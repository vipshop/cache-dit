"""Shared SVDQ module quantization flow and public all-precision entry."""
import torch
from torch import nn

from ...logger import init_logger
from ...utils import check_quantized

logger = init_logger(__name__)


def _quantize_svdq_wNaN(module: nn.Module, quantize_config, quantize_linear,
                        precision: str) -> nn.Module:
  """Share PTQ/DQ, calibration and checkpoint flow across SVDQ weight/activation formats.

  :param module: Root module containing the float `nn.Linear` layers to quantize.
  :param quantize_config: SVDQ config whose precision equals `precision`.
  :param quantize_linear: Format-specific function that packs one linear layer.
  :param precision: Manifest/runtime precision such as `fp8`, `w4a8` or `w2a4`.
  :returns: The module with eligible linears replaced by their packed runtime modules.
  """
  if not isinstance(module, nn.Module):
    raise TypeError(f"Expected nn.Module, got {type(module)}.")
  if not quantize_config.is_svdq() or quantize_config.get_svdq_precision() != precision:
    raise ValueError(f"SVDQ {precision} quantization requires an svdq_{precision}_* config, got "
                     f"{quantize_config.quant_type!r}.")
  if check_quantized(module):
    logger.warning("Module %s is already quantized, skipping SVDQ-%s quantization.",
                   module.__class__.__name__, precision.upper())
    return module
  from .ptq import SVDQPTQContext
  from .ptq import _attach_quantization_metadata
  from .ptq import _build_quant_config_snapshot
  from .ptq import _call_with_supported_kwargs
  from .ptq import _get_named_submodule
  from .ptq import _get_parent_and_child
  from .ptq import _log_quantize_summary
  from .ptq import _maybe_collect_svdq_garbage
  from .ptq import _resolve_svdq_quantize_device
  from .ptq import _save_quant_config_snapshot
  from .ptq import _save_quantized_module

  context = SVDQPTQContext.from_config(module, quantize_config)
  kwargs = context.svdq_kwargs
  quantize_device = _resolve_svdq_quantize_device(module, kwargs)
  is_dq = quantize_config.is_svdq_dq()
  strategy = "activation" if not is_dq else kwargs.get("smooth_strategy", "identity")
  if strategy == "few_shot":
    raise ValueError(f"SVDQ-{precision} does not support the few_shot smooth strategy.")
  calibrate_precision = kwargs.get("calibrate_precision", "low")
  format_kwargs = {}
  if precision in ("w1a4", "w2a4"):
    format_kwargs["stages"] = int(kwargs["packed_a4_stages"]) if "packed_a4_stages" in kwargs else 3
    format_kwargs["inner_group"] = (int(kwargs["packed_a4_inner_group"])
                                    if "packed_a4_inner_group" in kwargs else 4)

  captured: dict[str, torch.Tensor] = {}
  if not is_dq:
    if quantize_config.calibrate_fn is None:
      raise ValueError(f"SVDQ-{precision} PTQ requires QuantizeConfig.calibrate_fn; use the `_dq` "
                       "variant for data-free quantization.")

    def make_hook(layer_name: str):

      def hook(_layer, args):
        if args and isinstance(args[0], torch.Tensor):
          value = args[0].detach()
          span = value.abs().reshape(-1, value.shape[-1]).amax(dim=0)
          captured[layer_name] = span if layer_name not in captured else torch.maximum(
            captured[layer_name], span)

      return hook

    handles = [
      _get_named_submodule(module, name).register_forward_pre_hook(make_hook(name))
      for name in context.candidate_layer_names
    ]
    try:
      _call_with_supported_kwargs(quantize_config.calibrate_fn, module=module, model=module)
    finally:
      for handle in handles:
        handle.remove()

  result = module
  quantized_layer_names: list[str] = []
  for name in context.candidate_layer_names:
    if not is_dq and captured.get(name) is None:
      context._record_skip(name, "not observed during calibration")
      continue
    linear = _get_named_submodule(module, name)
    packed = quantize_linear(linear,
                             captured.get(name),
                             rank=context.rank,
                             smooth_strategy=strategy,
                             torch_dtype=linear.weight.dtype,
                             device=quantize_device,
                             calibrate_precision=calibrate_precision,
                             **format_kwargs)
    if name == "":
      result = packed
    else:
      parent, child = _get_parent_and_child(module, name)
      setattr(parent, child, packed)
    quantized_layer_names.append(name)

  if not quantized_layer_names:
    raise RuntimeError(f"SVDQ-{precision} found no observed layers to quantize. "
                       f"Skipped layers: {context.skipped_reasons}.")
  _attach_quantization_metadata(
    result,
    quant_type=quantize_config.quant_type,
    quantized_layer_names=quantized_layer_names,
    exclude_layers=context.exclude_layers,
    svdq_kwargs=context.svdq_kwargs,
    checkpoint_path=quantize_config.serialize_to,
    quantize_config=quantize_config,
  )
  if quantize_config.serialize_to is not None:
    writer_rank = _save_quantized_module(
      result,
      serialize_to=quantize_config.serialize_to,
      quant_type=quantize_config.quant_type,
      rank=context.rank,
      precision=context.precision,
      quantized_layer_names=quantized_layer_names,
      svdq_kwargs=context.svdq_kwargs,
    )
    if writer_rank:
      _save_quant_config_snapshot(
        _build_quant_config_snapshot(
          checkpoint_path=quantize_config.serialize_to,
          quant_type=quantize_config.quant_type,
          rank=context.rank,
          exclude_layers=context.exclude_layers,
          regional_quantize=context.regional_quantize,
          repeated_blocks=context.repeated_blocks,
          verbose=context.verbose,
          svdq_kwargs=context.svdq_kwargs,
          backend=quantize_config.backend,
        ),
        checkpoint_path=quantize_config.serialize_to,
      )
  _log_quantize_summary(
    context,
    quantized_layer_names=quantized_layer_names,
    serialize_to=quantize_config.serialize_to,
    observed_layer_names=sorted(captured) if not is_dq else None,
  )
  _maybe_collect_svdq_garbage(quantize_device)
  return result


def quantize_svdq_wNaN(module: nn.Module, quantize_config) -> nn.Module:
  """Quantize a module through the single public all-precision SVDQ entry point.

  :param module: Root module containing float `nn.Linear` layers.
  :param quantize_config: An SVDQ quantization config.
  :returns: The module with eligible layers replaced by their SVDQ runtime modules.
  """
  precision = quantize_config.get_svdq_precision()
  if precision in ("int4", "nvfp4", "w4a4", "w1a4", "w2a4", "w3a4"):
    from .wNa4 import _get_svdq_wNa4_quantizer
    quantize_module = _get_svdq_wNa4_quantizer(precision)
  elif precision in ("fp8", "w8a8", "w4a8"):
    from .wNa8 import _get_svdq_wNa8_quantizer
    quantize_module = _get_svdq_wNa8_quantizer(precision)
  elif precision in ("w6a6", "fp6"):
    from .w6a6 import _get_svdq_w6a6_quantizer
    quantize_module = _get_svdq_w6a6_quantizer(precision)
  else:
    raise ValueError(f"quantize_svdq_wNaN does not accept {quantize_config.quant_type!r}.")

  return quantize_module(module, quantize_config)


__all__ = ["quantize_svdq_wNaN"]
