from enum import Enum


class ParallelismBackend(Enum):
    NATIVE_DIFFUSER = "Native_Diffuser"
    NATIVE_PYTORCH = "Native_PyTorch"
    NONE = "None"

    @classmethod
    def is_supported(cls, backend: "ParallelismBackend") -> bool:
        if backend in [cls.NATIVE_PYTORCH]:
            return True
        # Now, only Native_Diffuser backend is supported
        if backend in [cls.NATIVE_DIFFUSER]:
            try:
                import diffusers  # noqa: F401
            except ImportError:
                return False
            return True
        return False
