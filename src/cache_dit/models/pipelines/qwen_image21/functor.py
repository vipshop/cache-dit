"""Cache-dit patch functor for the fused Qwen-Image 2.1 transformer.

`QwenImage21PatchFunctor` rewrites both `transformer.forward` and every `block.forward`,
so a fused block forward installed before `enable_cache` would simply be overwritten.
This subclass runs the base patching first and then rebinds the fused block forward on
the real blocks, which is the only ordering that survives `enable_cache`,
`enable_parallelism` and the compile step that follows them.

`is_from_diffusers` is overridden because the base check inspects
``transformer.__module__``: ``_QwenImage21Transformer2DModel`` lives in cache-dit, so the
inherited check would skip patching entirely — silently, leaving every layer to read and
write layer 0's KV cache and collapsing the image to noise.
"""

import torch
from diffusers import QwenImage21Transformer2DModel

from ....caching.patch_functors.functor_qwen_image_21 import QwenImage21PatchFunctor
from ....logger import init_logger
from .block import install_fused_modules

logger = init_logger(__name__)


class _QwenImage21PatchFunctor(QwenImage21PatchFunctor):
  """Install the fused block forward on top of the stock Qwen-Image 2.1 patching."""

  @classmethod
  def is_from_diffusers(cls, transformer: torch.nn.Module) -> bool:
    """Accept the cache-dit subclass even though its module is not `diffusers`.

    :param transformer: Transformer about to be patched.
    :returns: Whether this functor supports the transformer.
    """

    return isinstance(transformer, QwenImage21Transformer2DModel)

  def _apply(
    self,
    transformer: QwenImage21Transformer2DModel,
    blocks: torch.nn.ModuleList = None,
    **kwargs,
  ) -> QwenImage21Transformer2DModel:
    """Run the stock patch, then rebind the fused block forward.

    :param transformer: Transformer to patch.
    :param blocks: Ignored, kept for signature compatibility.
    :param kwargs: Forwarded to the base implementation.
    :returns: The patched transformer.
    """

    transformer = super()._apply(transformer, blocks=blocks, **kwargs)

    install_fused_modules(transformer)

    if getattr(transformer, "_old_forward", None) is not None:
      # accelerate hooks captured the pre-patch forward and `mock_transformer` prefers
      # `_old_forward` when `_hf_hook` is set, so it has to follow every rewrite.
      transformer._old_forward = transformer.forward

    transformer._is_qwen21_fused = True
    return transformer


__all__ = ["_QwenImage21PatchFunctor"]
