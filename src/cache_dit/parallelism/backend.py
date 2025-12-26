from enum import Enum


class ParallelismBackend(Enum):
    NATIVE_DIFFUSER = "Native_Diffuser"
    NATIVE_PYTORCH = "Native_PyTorch"
    NONE = "None"

    @classmethod
    def is_supported(cls, backend: "ParallelismBackend") -> bool:
        if backend == cls.NATIVE_PYTORCH:
            return True
        elif backend == cls.NATIVE_DIFFUSER:
            try:
                from diffusers.models._modeling_parallel import (  # noqa F401
                    ContextParallelModelPlan,
                )
            except ImportError:
                raise ImportError(
                    "NATIVE_DIFFUSER parallelism backend requires the latest "
                    "version of diffusers(>=0.36.dev0). Please install latest "
                    "version of diffusers from source: \npip3 install "
                    "git+https://github.com/huggingface/diffusers.git"
                )
            return True
        return False

    def __str__(self) -> str:
        return self.value
