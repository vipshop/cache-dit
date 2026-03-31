from enum import Enum


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
        else:
            # Only Triton backend is supported for now, we can add more
            # backends in the future.
            return False
