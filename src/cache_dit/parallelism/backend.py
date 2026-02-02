from enum import Enum


def _check_diffusers_cp_support():
    try:
        from diffusers.models._modeling_parallel import (  # noqa F401
            ContextParallelModelPlan,
        )
    except ImportError:
        raise ImportError(
            "context parallelism backend requires the latest "
            "version of diffusers(>=0.36.dev0). Please install latest "
            "version of diffusers from source: \npip3 install "
            "git+https://github.com/huggingface/diffusers.git"
        )
    return True


class ParallelismBackend(Enum):
    AUTO = "Auto"
    NATIVE_DIFFUSER = "Native_Diffuser"  # CP/SP
    NATIVE_PYTORCH = "Native_PyTorch"  # TP or DP
    NATIVE_HYBRID = "Native_Hybrid"  # CP/SP + TP
    NONE = "None"

    @classmethod
    def is_supported(cls, backend: "ParallelismBackend") -> bool:
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
        for backend in cls:
            if backend.value.lower() == backend_str.lower():
                return backend
        raise ValueError(f"Unsupported parallelism backend: {backend_str}.")

    def __str__(self) -> str:
        return self.value
