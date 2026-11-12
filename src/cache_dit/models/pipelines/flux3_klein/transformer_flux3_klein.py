"""FLUX.3 Klein transformer with cache-dit's memory-bound fused operators.

The class only adds behaviour: `__init__`, every submodule name and therefore every
state-dict key stay exactly as diffusers defines them, which is what keeps
`from_pretrained`, the modular pipeline's component loading, tensor parallelism
(`_tp_plan` uses dotted module paths) and checkpoint compatibility untouched.

FLUX.3 is a `ModularPipeline`, so there is no `_Flux3KleinPipeline` subclass and the upgrade
happens on the transformer only: the CLI example's `post_init_hook` swaps
`pipe.transformer.__class__` and calls :meth:`apply_fused_ops` right after the weights are
loaded, i.e. before `enable_cache`, `maybe_compile_transformer` and any offload hook.

`forward` is deliberately **not** overridden. `diffusers/modular_pipelines/flux3/denoise.py`
decides which inputs to pass by inspecting `inspect.signature(transformer.forward)`, so a
rewritten forward with a different signature silently drops `img_ids` / `txt_ids` and the
model then runs without RoPE.
"""

import torch
from diffusers import Flux3Transformer2DModel

from ....logger import init_logger
from .block import fused_modules_installed, install_fused_modules, restore_fused_modules

logger = init_logger(__name__)


class _Flux3KleinTransformer2DModel(Flux3Transformer2DModel):
  """Diffusers' FLUX.3 transformer with fused block and attention seams.

  Supports the same `forward` contract, the same `_repeated_blocks` contract and the same
  `_keep_in_fp32_modules` set (the q/k norm weights stay fp32, which the fused kernel expects).
  """

  def apply_fused_ops(self) -> None:
    """Bind the fused block forwards and attention processors. Safe to call more than once.

    Only instance attributes are rebound, so the block class names still match
    `_repeated_blocks` and cache-dit's cached-blocks wrapper keeps driving the same modules.
    """

    if getattr(self, "_flux3_klein_fused_installed", False):
      return

    install_fused_modules(self)

    logger.info("FLUX.3 Klein fused ops installed on "
                f"{len(self.single_transformer_blocks)} joint blocks and "
                f"{len(self.txt_mode_blocks) + sum(len(s) for s in self.mode_blocks.values())} "
                "per-mode blocks.")

  def remove_fused_ops(self) -> None:
    """Restore the stock forwards and processors, for parity tests and A/B runs."""

    if not getattr(self, "_flux3_klein_fused_installed", False):
      return
    restore_fused_modules(self)
    logger.info("FLUX.3 Klein fused ops removed.")


def upgrade_transformer(transformer: torch.nn.Module) -> torch.nn.Module:
  """Swap a loaded `Flux3Transformer2DModel` for the fused subclass, in place.

  A `__class__` swap keeps the loaded weights, hooks and quantization wrappers: only the
  Python behaviour changes, so nothing has to be re-loaded or moved.

  :param transformer: Transformer owned by the pipeline.
  :returns: The upgraded transformer, or the input when it is not a FLUX.3 transformer.
  """

  if not isinstance(transformer, Flux3Transformer2DModel):
    logger.warning(f"Expected a Flux3Transformer2DModel, got {type(transformer).__name__}; "
                   "fused ops are not installed.")
    return transformer

  if not isinstance(transformer, _Flux3KleinTransformer2DModel):
    transformer.__class__ = _Flux3KleinTransformer2DModel

  transformer.apply_fused_ops()
  return transformer


__all__ = [
  "_Flux3KleinTransformer2DModel",
  "fused_modules_installed",
  "upgrade_transformer",
]
