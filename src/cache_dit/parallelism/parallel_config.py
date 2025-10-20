import dataclasses
from cache_dit.parallelism.parallel_backend import ParallelismBackend
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class ParallelismConfig:
    # Parallelism backend, defaults to NATIVE_DIFFUSER
    backend: ParallelismBackend = ParallelismBackend.NATIVE_DIFFUSER
    # Context parallelism config
    # ulysses_size (`int`, *optional*):
    #     The degree of ulysses parallelism.
    ulysses_size: int = None
    # ring_size (`int`, *optional*):
    #     The degree of ring parallelism.
    ring_size: int = None
    # Tensor parallelism config
    # tp_size (`int`, *optional*):
    #     The degree of tensor parallelism.
    tp_size: int = None

    def __post_init__(self):
        assert ParallelismBackend.is_supported(self.backend), (
            f"Parallel backend {self.backend} is not supported. "
            f"Please make sure the required packages are installed."
        )
        assert self.tp_size is None, "Tensor parallelism is not supported yet."

    def strify(self, details: bool = False) -> str:
        if details:
            return (
                f"ParallelismConfig(backend={self.backend}, "
                f"ulysses_size={self.ulysses_size}, "
                f"ring_size={self.ring_size}, "
                f"tp_size={self.tp_size})"
            )
        else:
            parallel_str = ""
            if self.ulysses_size is not None:
                parallel_str += f"Ulysses{self.ulysses_size}"
            if self.ring_size is not None:
                parallel_str += f"Ring{self.ring_size}"
            if self.tp_size is not None:
                parallel_str += f"TP{self.tp_size}"
            return parallel_str
