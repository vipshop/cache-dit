from enum import Enum


class QuantizeBackend(Enum):
    AUTO = "Auto"
    TORCHAO = "TorchAo"  # PyTorch TorchAo backend
    CACHE_DIT = "CACHE_DIT"  # Native quantization support in Cache-DiT
    NONE = "None"

    @classmethod
    def from_str(cls, backend_str: str) -> "QuantizeBackend":
        alias_map = {
            "ao": cls.TORCHAO,
            "torchao": cls.TORCHAO,
            "cache_dit": cls.CACHE_DIT,
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
    def is_supported(cls, backend: "QuantizeBackend") -> bool:
        if backend == cls.AUTO:
            return True
        elif backend == cls.TORCHAO:
            try:
                import torchao  # noqa F401

                return True
            except ImportError:
                return False
        elif backend == cls.CACHE_DIT:
            # Native quantization support in Cache-DiT is not available yet, we will
            # support it in the future.
            return False
        elif backend == cls.NONE:
            raise ValueError("QuantizeBackend.NONE is not a valid backend")
        return False
