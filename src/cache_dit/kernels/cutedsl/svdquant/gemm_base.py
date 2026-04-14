from __future__ import annotations

import os
from dataclasses import dataclass

_LOGICAL_BLOCK_M_ENV = "CACHE_DIT_SVDQ_V2_BLOCK_M"
_SUPPORTED_BLOCK_M = (64, 128, 256)


@dataclass(frozen=True)
class SVDQW4A4V3Config:
  """Runtime configuration shared by the SVDQ v3 CuTe DSL wrappers.

  The real CuTe DSL kernels will eventually specialize on these fields during JIT compilation. The
  compatibility layer records them now so the surrounding Python call path and future cache-key
  design stay stable.
  """

  stage: int = 1
  logical_block_m: int | None = None
  fp4: bool = False
  act_unsigned: bool = False

  @classmethod
  def from_env(
    cls,
    *,
    stage: int,
    fp4: bool,
    act_unsigned: bool,
  ) -> "SVDQW4A4V3Config":
    env_value = os.environ.get(_LOGICAL_BLOCK_M_ENV)
    if env_value is None:
      logical_block_m = None
    else:
      logical_block_m = int(env_value)
      if logical_block_m not in _SUPPORTED_BLOCK_M:
        raise ValueError(
          f"{_LOGICAL_BLOCK_M_ENV} must be one of {_SUPPORTED_BLOCK_M}, got {logical_block_m}.")
    return cls(
      stage=stage,
      logical_block_m=logical_block_m,
      fp4=fp4,
      act_unsigned=act_unsigned,
    )


__all__ = ["SVDQW4A4V3Config"]
