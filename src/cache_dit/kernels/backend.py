from enum import Enum

import torch


class KernelBackend(Enum):
  TRITON = "Triton"
  CUDA = "CUDA"
  CUTEDSL = "CuteDSL"
  NONE = "None"

  @classmethod
  def from_str(cls, backend_str: str) -> "KernelBackend":
    for backend in cls:
      if backend.value.lower() == backend_str.lower():
        return backend
    raise ValueError(f"Unsupported kernel backend: {backend_str}.")

  @classmethod
  def is_supported(cls, backend: "KernelBackend") -> bool:
    if backend == cls.TRITON:
      try:
        import triton  # noqa F401

        return True
      except ImportError:
        return False
    if backend == cls.CUDA:
      if not torch.cuda.is_available():
        return False
      # The SVDQuant CUDA extension is currently required for the CUDA backend,
      # but we check for it at runtime in the kernels, so we return True here as
      # long as CUDA is available.
      return True
    if backend == cls.CUTEDSL:
      if not torch.cuda.is_available():
        return False
      try:
        import cutlass.cute  # noqa F401

        return True
      except ImportError:
        return False
    return False
