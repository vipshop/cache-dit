from enum import Enum


def _check_diffusers_cp_support():
  """Validate the diffusers APIs required by cache-dit's CP runtime.

  The cache-dit CP runtime no longer depends on diffusers' private
  `_modeling_parallel` module. Instead, we validate the stable modeling and
  attention symbols used by cache-dit's own CP bridge/runtime.
  """

  try:
    from diffusers.models.attention_processor import Attention  # noqa F401
    from diffusers.models.modeling_utils import ModelMixin  # noqa F401
  except ImportError as exc:
    raise ImportError(
      "context parallelism backend requires diffusers modeling and attention APIs used by "
      "cache-dit. Please install a recent diffusers version from source: \npip3 install "
      "git+https://github.com/huggingface/diffusers.git") from exc

  try:
    from diffusers.models.attention import AttentionModuleMixin  # noqa F401
  except ImportError:
    pass
  return True


class ParallelismBackend(Enum):
  """Enumerate the parallel execution backends supported by cache-dit."""

  AUTO = "Auto"
  NATIVE_DIFFUSER = "Native_Diffuser"  # CP/SP
  NATIVE_PYTORCH = "Native_PyTorch"  # TP or DP
  NATIVE_HYBRID = "Native_Hybrid"  # CP/SP + TP
  NONE = "None"

  @classmethod
  def is_supported(cls, backend: "ParallelismBackend") -> bool:
    """Return whether a backend is available in the current environment.

    :param backend: Backend enum value to validate.
    :returns: `True` when the backend can be used on the current installation.
    """

    if backend == cls.AUTO:
      return True
    elif backend == cls.NATIVE_PYTORCH:
      return True
    elif backend == cls.NATIVE_DIFFUSER:
      return _check_diffusers_cp_support()
    elif backend == cls.NATIVE_HYBRID:
      return _check_diffusers_cp_support()
    elif backend == cls.NONE:
      raise ValueError("ParallelismBackend.NONE is not a valid backend")
    return False

  @classmethod
  def from_str(cls, backend_str: str) -> "ParallelismBackend":
    """Parse a user-facing backend string into a `ParallelismBackend` value.

    :param backend_str: Backend name supplied by the user or config.
    :returns: The matching `ParallelismBackend` enum value.
    """

    for backend in cls:
      if backend.value.lower() == backend_str.lower():
        return backend
    raise ValueError(f"Unsupported parallelism backend: {backend_str}.")

  def __str__(self) -> str:
    return self.value
