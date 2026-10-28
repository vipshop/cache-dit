"""Runtime patches for upstream model code that cache-dit has to correct.

Every patch here is inert until the caller opts in (an ENV switch or an explicit call), because a
patch is a copy of upstream code and has to be re-synced whenever upstream changes. Keep the list
short and prefer deleting an entry once upstream ships the fix.
"""
import inspect
from typing import Optional

import torch
import torch.nn.functional as F

from .envs import ENV
from .logger import init_logger

logger = init_logger(__name__)

_PATCH_MARKER = "_cache_dit_patched"

_NUCLEUS_TRANSFORMER_NAME = "NucleusMoEImageTransformer2DModel"
_OFFSETS_ENV_HINT = ("CACHE_DIT_NUCLEUS_MOE_GROUPED_MM_OFFSETS_FIX=1 (set it before importing "
                     "cache-dit) or call `cache_dit.patch_nucleus_moe_grouped_mm_offsets()`")


def _nucleus_experts_cls() -> Optional[type]:
  """Return diffusers' `SwiGLUExperts`, or None when the installed version does not ship it."""
  try:
    from diffusers.models.transformers.transformer_nucleusmoe_image import SwiGLUExperts
  except ImportError:
    return None
  return SwiGLUExperts


def patch_nucleus_moe_grouped_mm_offsets() -> bool:
  """Make `SwiGLUExperts._run_experts_grouped_mm` emit int32 offsets under `torch.compile`.

  Upstream computes the grouped-GEMM offsets with
  `torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)`. Inductor's `cumsum`
  decomposition drops the `dtype` argument, so inside a compiled graph the offsets come out as
  int64 and `aten._grouped_mm` rejects them ("Offsets tensor must be integer (int32) tensor, but
  got torch.int64"). Casting *after* the cumsum leaves an explicit `aten.to` node in the graph
  that survives compilation, and the values are identical for any token count below 2**31.

  Adapted from diffusers' `SwiGLUExperts._run_experts_grouped_mm`; the copy is only installed while
  the upstream source still matches the shape it was taken from, so an upstream rewrite cannot
  silently pin old semantics.

  :returns: True if the patch was installed, False if it was already installed, the model is absent
    from the installed diffusers, or upstream no longer looks like the code this copy was taken
    from.
  """
  experts_cls = _nucleus_experts_cls()
  if experts_cls is None:
    logger.warning_once("The installed diffusers does not provide `SwiGLUExperts`, so the "
                        "grouped-GEMM offsets patch is skipped.")
    return False

  upstream = getattr(experts_cls, "_run_experts_grouped_mm", None)
  if upstream is None:
    logger.warning_once("The installed diffusers has no `SwiGLUExperts._run_experts_grouped_mm`, "
                        "so the grouped-GEMM offsets patch is skipped.")
    return False
  if getattr(upstream, _PATCH_MARKER, False):
    return False
  if "dtype=torch.int32" not in inspect.getsource(upstream):
    # The installed version either already narrows the offsets itself or was rewritten; either way
    # this copy must not be applied.
    logger.warning_once(
      "`SwiGLUExperts._run_experts_grouped_mm` no longer matches the upstream "
      "code this patch was taken from, so the patch is skipped. Please report it.")
    return False

  def _run_experts_grouped_mm(self, x: torch.Tensor,
                              num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """Upstream body with the offset narrowing moved after the cumsum."""
    offsets = torch.cumsum(num_tokens_per_expert, dim=0).to(torch.int32)

    gate_up = F.grouped_mm(x, self.gate_up_proj, offs=offsets)
    gate, up = gate_up.chunk(2, dim=-1)
    out = F.grouped_mm(F.silu(gate) * up, self.down_proj, offs=offsets)

    return out.type_as(x)

  setattr(_run_experts_grouped_mm, _PATCH_MARKER, True)
  experts_cls._run_experts_grouped_mm = _run_experts_grouped_mm
  logger.info("Patched `SwiGLUExperts._run_experts_grouped_mm` to emit int32 grouped-GEMM offsets "
              "(required by torch.compile).")
  return True


def maybe_patch_nucleus_moe_grouped_mm_offsets(
    transformer: Optional[torch.nn.Module] = None) -> bool:
  """Apply the NucleusMoE workaround when the ENV switch asks for it.

  When the switch is off and `transformer` is a NucleusMoE transformer, log the one-line hint: that
  model cannot be compiled without the patch, and the raw `aten._grouped_mm` error does not say so.

  :param transformer: Transformer about to be compiled, used only to detect NucleusMoE-Image.
  :returns: True if the patch is installed.
  """
  if ENV.CACHE_DIT_NUCLEUS_MOE_GROUPED_MM_OFFSETS_FIX:
    return patch_nucleus_moe_grouped_mm_offsets()

  if transformer is not None and type(transformer).__name__ == _NUCLEUS_TRANSFORMER_NAME:
    logger.warning_once(
      "Compiling %s fails inside `aten._grouped_mm` unless the offsets patch is "
      "installed. Enable it with %s.", _NUCLEUS_TRANSFORMER_NAME, _OFFSETS_ENV_HINT)
  return False
