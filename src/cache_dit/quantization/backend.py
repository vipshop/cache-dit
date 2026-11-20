from enum import Enum


class QuantizeBackend(Enum):
  AUTO = "AUTO"
  TORCHAO = "TORCHAO"  # PyTorch TorchAo backend
  CACHE_DIT = "CACHE_DIT"  # Native quantization support in Cache-DiT
  NONE = "NONE"

  @classmethod
  def from_str(cls, backend_str: str) -> "QuantizeBackend":
    alias_map = {
      "ao": cls.TORCHAO,
      "torchao": cls.TORCHAO,
      "cache_dit": cls.CACHE_DIT,
      "cache-dit": cls.CACHE_DIT,
      "cachedit": cls.CACHE_DIT,
      "auto": cls.AUTO,
      "none": cls.NONE,
    }
    backend_str_lower = backend_str.lower()
    if backend_str_lower in alias_map:
      return alias_map[backend_str_lower]
    for backend in cls:
      if backend.value.lower() == backend_str_lower:
        return backend
    raise ValueError(f"Unsupported quantization backend: {backend_str}.")

  @classmethod
  def is_supported(cls,
                   backend: "QuantizeBackend",
                   quant_type: str | None = None,
                   svdq_kwargs: dict | None = None) -> bool:
    """Check whether a backend can serve the given quant type on this host.

    :param backend: Resolved quantization backend.
    :param quant_type: Requested quant type; the native CACHE_DIT backend needs
      it because SVDQuant/native FP8/native NVFP4 require the ``_C_ops`` CUDA
      extension while MXFP4 only needs the MX dtypes from PyTorch.
    :param svdq_kwargs: Raw SVDQuant kwargs; ``runtime_kernel`` selects a
      self-contained runtime (FlashInfer / cute) that does not need ``_C_ops``.
    :returns: Whether the config can be quantized on this host.
    """
    if backend == cls.AUTO:
      return True
    elif backend == cls.TORCHAO:
      try:
        import torchao  # noqa F401

        return True
      except ImportError:
        return False
    elif backend == cls.CACHE_DIT:
      quant_type = str(quant_type or "").lower()
      if quant_type.startswith("mxfp4"):
        try:
          from .mxfp4 import mxfp4_is_available

          return mxfp4_is_available()
        except ImportError:
          return False
      runtime = str((svdq_kwargs or {}).get("runtime_kernel") or "").lower()
      if runtime.startswith("flashinfer") or runtime == "cute":
        # These SVDQuant runtimes ship their own kernels and never import the
        # unified `_C_ops` extension; their dependencies are validated when the
        # runtime is built, so do not gate them on the native extension here.
        return True
      try:
        from ..kernels import cuda_ext_is_available

        return cuda_ext_is_available()
      except ImportError:
        return False
    elif backend == cls.NONE:
      raise ValueError("QuantizeBackend.NONE is not a valid backend")
    return False
