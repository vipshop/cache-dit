from enum import Enum

import torch


class KernelBackend(Enum):
  CUDA = "CUDA"
  MINDIESD = "MindIESD"
  NONE = "None"

  @classmethod
  def from_str(cls, backend_str: str) -> "KernelBackend":
    for backend in cls:
      if backend.value.lower() == backend_str.lower():
        return backend
    raise ValueError(f"Unsupported kernel backend: {backend_str}.")

  @classmethod
  def is_supported(cls, backend: "KernelBackend") -> bool:
    if backend == cls.CUDA:
      return torch.cuda.is_available()
    if backend == cls.MINDIESD:
      try:
        import mindiesd  # noqa F401

        return True
      except Exception:
        return False
    return False
