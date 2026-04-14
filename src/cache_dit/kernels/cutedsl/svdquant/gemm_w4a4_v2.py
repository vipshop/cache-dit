from __future__ import annotations

import torch

from ...cuda import svdq_gemm_w4a4_v2 as _cuda_svdq_gemm_w4a4_v2
from .gemm_base import SVDQW4A4V3Config
from .gemm_utils import normalize_runtime_stage
from .gemm_utils import require_int4_runtime
from .gemm_utils import should_allow_cuda_fallback


def svdq_gemm_w4a4_v2(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None = None,
  lora_up: torch.Tensor | None = None,
  bias: torch.Tensor | None = None,
  fp4: bool = False,
  alpha: float | None = 1.0,
  wcscales: torch.Tensor | None = None,
  act_unsigned: bool = False,
  output_dtype: torch.dtype | None = None,
  stage: int = 1,
) -> torch.Tensor:
  """Dispatch the SVDQ runtime v3 GEMM wrapper.

  The final CuTe DSL implementation will specialize on the same packed INT4
  contract currently used by `svdq_gemm_w4a4_v2`. The initial scaffold keeps
  the public interface intact and records the future configuration boundary
  while the mainloop implementation is still being ported.
  """

  require_int4_runtime(fp4, "svdq_gemm_w4a4_v2")
  normalized_stage = normalize_runtime_stage(stage, "svdq_gemm_w4a4_v2")
  _ = SVDQW4A4V3Config.from_env(
    stage=normalized_stage,
    fp4=fp4,
    act_unsigned=act_unsigned,
  )
  if not should_allow_cuda_fallback():
    raise NotImplementedError(
      "svdq_gemm_w4a4_v2 v3 kernel scaffolding is present, but the CuTe DSL implementation "
      "has not been wired yet.")
  return _cuda_svdq_gemm_w4a4_v2(
    act=act,
    wgt=wgt,
    ascales=ascales,
    wscales=wscales,
    lora_act_in=lora_act_in,
    lora_up=lora_up,
    bias=bias,
    fp4=fp4,
    alpha=alpha,
    wcscales=wcscales,
    act_unsigned=act_unsigned,
    output_dtype=output_dtype,
    stage=normalized_stage,
  )


__all__ = ["svdq_gemm_w4a4_v2"]
