"""Qwen-Image 2.1 transformer with cache-dit's memory-bound fused operators.

The class only adds behaviour: `__init__`, every submodule name and therefore every
state-dict key stay exactly as diffusers defines them, which is what keeps
`from_pretrained`, tensor parallelism (`_tp_plan` uses dotted module paths), layerwise
casting patterns and checkpoint compatibility untouched.

Install is idempotent and deliberately does *not* touch `_is_patched`,
`_qwenimage21_blocks` or `block._idx`: those belong to the patch functor. Writing
`_is_patched` here would make `enable_cache` skip its own patch and silently collapse the
per-layer KV cache — see :mod:`.functor`.
"""

import torch
from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21FlexAttnProcessor

from ....logger import init_logger
from .block import install_fused_modules
from .fused_ops import probe_ln_tier

logger = init_logger(__name__)


class _QwenImage21Transformer2DModel(QwenImage21Transformer2DModel):
  """Diffusers' Qwen-Image 2.1 transformer with fused block forwards.

  Supports the same `forward` contract, including the per-layer prefix KV cache
  (`kv_cache` + `kv_cache_mode`) that the pipeline drives.
  """

  def apply_fused_ops(self) -> None:
    """Bind the fused block forward on every block. Safe to call more than once.

    Only the instance's `forward` is rebound, so the block class name still matches
    `_repeated_blocks = ["QwenImage21TransformerBlock"]` and the cached-blocks wrapper
    keeps driving the same modules.
    """

    if getattr(self, "_qwen21_fused_installed", False):
      return

    install_fused_modules(self)

    # Bit-exactness cannot be assumed: ATen's reduction layout is a property of the kernel
    # it selects on this device, and which product nvcc folds into an FMA depends on the
    # build. Both probes run here, before `torch.compile`, so the compiled graph never sees
    # a probe (they sync the host) and the fallbacks are decided once per process.
    hidden_size = getattr(self.config, "attention_head_dim", 128) * getattr(
      self.config, "num_attention_heads", 32)
    device = next(self.parameters()).device if any(True for _ in self.parameters()) else None
    if device is not None and device.type == "cuda":
      probe_ln_tier(hidden_size, device)

    if getattr(self, "_old_forward", None) is not None:
      self._old_forward = self.forward

    flex_blocks = [
      index for index, block in enumerate(self.transformer_blocks)
      if isinstance(block.attn.processor, QwenImage21FlexAttnProcessor)
    ]
    if flex_blocks:
      logger.info(f"Qwen-Image 2.1 fused ops keep the flex attention processors untouched "
                  f"({len(flex_blocks)} blocks); their prefill mask is built by the pipeline.")

    self._qwen21_fused_installed = True
    logger.info(f"Qwen-Image 2.1 fused ops installed on {len(self.transformer_blocks)} blocks.")


def upgrade_transformer(transformer: torch.nn.Module) -> torch.nn.Module:
  """Swap a loaded `QwenImage21Transformer2DModel` for the fused subclass, in place.

  A `__class__` swap keeps the loaded weights, hooks and quantization wrappers: only the
  Python behaviour changes, so nothing has to be re-loaded or moved.

  :param transformer: Transformer owned by the pipeline.
  :returns: The upgraded transformer, or the input when it is not a Qwen-Image 2.1 model.
  """

  if not isinstance(transformer, QwenImage21Transformer2DModel):
    logger.warning(f"Expected a QwenImage21Transformer2DModel, got {type(transformer).__name__}; "
                   "fused ops are not installed.")
    return transformer

  if not isinstance(transformer, _QwenImage21Transformer2DModel):
    transformer.__class__ = _QwenImage21Transformer2DModel

  transformer.apply_fused_ops()
  return transformer


__all__ = ["_QwenImage21Transformer2DModel", "upgrade_transformer"]
