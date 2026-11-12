"""FLUX.2-Klein transformer with cache-dit's memory-bound fused operators.

The class only adds behaviour: `__init__`, every submodule name and therefore every state-dict
key stay exactly as diffusers defines them, which is what keeps `from_pretrained`, tensor
parallelism (dotted module paths), layerwise casting patterns and checkpoint compatibility
untouched.

`apply_fused_ops` is idempotent and installs two things:

* the fused block forwards (see :mod:`.block`),
* the fused attention processors (see :mod:`.attention`).

It can also install the rotation-table cache, the only change outside the kernels: `Flux2PosEmbed`
rebuilds the float64 `cos`/`sin` tables on every forward although `img_ids`/`txt_ids` do not change
between denoising steps. That cache measured neutral, so it currently ships disabled
(:data:`.fused_ops.ROPE_CACHE_ENABLED` is `False`).
"""

import torch
from diffusers import Flux2Transformer2DModel

from ....logger import init_logger
from .block import install_fused_modules

logger = init_logger(__name__)

_ROPE_CACHE_ATTRIBUTE = "_flux2_klein_pos_embed_cache"
_ROPE_CACHE_LIMIT = 4


def _ids_key(ids: torch.Tensor) -> tuple:
  """Build a cheap cache key for a position-id tensor.

  The pipeline builds `img_ids`/`txt_ids` once per `__call__` and hands the same tensor to
  every denoising step, so identity (pointer, shape, dtype, device) identifies the table. A
  different resolution or device produces a different key and recomputes.

  :param ids: Position ids, `[S, axes]`.
  :returns: Hashable key.
  """

  return (ids.data_ptr(), tuple(ids.shape), ids.dtype, ids.device)


def _install_rope_cache(pos_embed: torch.nn.Module) -> None:
  """Cache the rotation tables of one `Flux2PosEmbed` instance.

  The wrapper is `torch.compiler.disable`d on purpose: a dictionary lookup on a tensor pointer
  cannot be traced, and letting dynamo specialize on it would add guards for no benefit. The
  graph break sits before the block loop, so nothing fusable is split.

  :param pos_embed: The model's `Flux2PosEmbed` instance.
  """

  if getattr(pos_embed, _ROPE_CACHE_ATTRIBUTE, None) is not None:
    return

  original = pos_embed.forward
  cache: dict = {}
  setattr(pos_embed, _ROPE_CACHE_ATTRIBUTE, cache)
  setattr(pos_embed, "_flux2_klein_original_forward", original)

  @torch.compiler.disable
  def cached_forward(ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    key = _ids_key(ids)
    hit = cache.get(key)
    if hit is not None:
      return hit
    tables = original(ids)
    if len(cache) >= _ROPE_CACHE_LIMIT:
      cache.clear()
    cache[key] = tables
    return tables

  pos_embed.forward = cached_forward


def _remove_rope_cache(pos_embed: torch.nn.Module) -> None:
  """Restore the stock `Flux2PosEmbed.forward`.

  :param pos_embed: The model's `Flux2PosEmbed` instance.
  """

  original = getattr(pos_embed, "_flux2_klein_original_forward", None)
  if original is None:
    return
  pos_embed.forward = original
  delattr(pos_embed, "_flux2_klein_original_forward")
  setattr(pos_embed, _ROPE_CACHE_ATTRIBUTE, None)


class _Flux2KleinTransformer2DModel(Flux2Transformer2DModel):
  """Diffusers' FLUX.2 transformer with cache-dit's fused memory-bound operators."""

  def apply_fused_ops(self) -> None:
    """Bind the fused forwards and processors, plus the rotation cache when it is enabled.

    Safe to call twice.
    """

    from .fused_ops import ROPE_CACHE_ENABLED

    if getattr(self, "_flux2_klein_fused_installed", False):
      return

    install_fused_modules(self)
    if ROPE_CACHE_ENABLED:
      _install_rope_cache(self.pos_embed)

    if getattr(self, "_old_forward", None) is not None:
      # accelerate hooks captured the pre-patch forward, so it has to follow every rewrite.
      self._old_forward = self.forward

    self._flux2_klein_fused_installed = True
    logger.info(f"FLUX.2 Klein fused ops installed on {len(self.transformer_blocks)} double and "
                f"{len(self.single_transformer_blocks)} single blocks.")

  @property
  def fused_ops_installed(self) -> bool:
    """Whether :meth:`apply_fused_ops` has run on this instance."""

    return getattr(self, "_flux2_klein_fused_installed", False)

  def remove_fused_ops(self) -> None:
    """Undo the attention half of :meth:`apply_fused_ops`.

    Restores the stock processors and drops the rotation-table cache. The block forwards stay
    rebound on purpose - they are plain Python bound methods over the same tensors - so this does
    **not** stop the operators: every one of them still dispatches to its kernel. To switch the
    operators themselves off, set `CACHE_DIT_FUSED_OPS=0` before the first fused pipeline is built
    or call :func:`set_flux2_klein_fused_ops_enabled` (see :mod:`.fused_ops`).
    """

    from .attention import uninstall_attention_patch

    if not self.fused_ops_installed:
      return
    uninstall_attention_patch(self)
    _remove_rope_cache(self.pos_embed)
    self._flux2_klein_fused_installed = False


def upgrade_transformer(transformer: torch.nn.Module) -> torch.nn.Module:
  """Swap a loaded `Flux2Transformer2DModel` for the fused subclass, in place.

  A `__class__` swap keeps the loaded weights, hooks and quantization wrappers: only the Python
  behaviour changes, so nothing has to be re-loaded or moved.

  :param transformer: Transformer owned by the pipeline.
  :returns: The upgraded transformer, or the input when it is not a FLUX.2 transformer.
  """

  if transformer is None:
    return transformer
  if not isinstance(transformer, Flux2Transformer2DModel):
    logger.warning(f"Expected a Flux2Transformer2DModel, got {type(transformer).__name__}; "
                   "fused ops are not installed.")
    return transformer

  if not isinstance(transformer, _Flux2KleinTransformer2DModel):
    transformer.__class__ = _Flux2KleinTransformer2DModel

  transformer.apply_fused_ops()
  return transformer


__all__ = ["_Flux2KleinTransformer2DModel", "upgrade_transformer"]
