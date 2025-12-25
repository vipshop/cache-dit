import dataclasses
from typing import Optional, Dict, Any
from cache_dit.parallelism.backend import ParallelismBackend
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
    # parallel_kwargs (`dict`, *optional*):
    #     Additional kwargs for parallelism backends. For example, for
    #     NATIVE_DIFFUSER backend, it can include `cp_plan` and
    #     `attention_backend` arguments for `Context Parallelism`.
    parallel_kwargs: Optional[Dict[str, Any]] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        assert ParallelismBackend.is_supported(self.backend), (
            f"Parallel backend {self.backend} is not supported. "
            f"Please make sure the required packages are installed."
        )

        # Validate the parallelism configuration and auto adjust the backend if needed
        if self.tp_size is not None and self.tp_size > 1:
            assert (
                self.ulysses_size is None or self.ulysses_size == 1
            ), "Tensor parallelism plus Ulysses parallelism is not supported right now."
            assert (
                self.ring_size is None or self.ring_size == 1
            ), "Tensor parallelism plus Ring parallelism is not supported right now."
            if self.backend != ParallelismBackend.NATIVE_PYTORCH:
                logger.warning(
                    "Tensor parallelism is only supported for NATIVE_PYTORCH backend "
                    "right now. Force set backend to NATIVE_PYTORCH."
                )
                self.backend = ParallelismBackend.NATIVE_PYTORCH
        elif (
            self.ulysses_size is not None
            and self.ulysses_size > 1
            and self.ring_size is not None
            and self.ring_size > 1
        ):
            raise ValueError(
                "Ulysses parallelism plus Ring parallelism is not fully supported right now."
            )
        else:
            if (self.ulysses_size is not None and self.ulysses_size > 1) or (
                self.ring_size is not None and self.ring_size > 1
            ):
                if self.backend != ParallelismBackend.NATIVE_DIFFUSER:
                    logger.warning(
                        "Ulysses/Ring parallelism is only supported for NATIVE_DIFFUSER "
                        "backend right now. Force set backend to NATIVE_DIFFUSER."
                    )
                    self.backend = ParallelismBackend.NATIVE_DIFFUSER

    def strify(
        self,
        details: bool = False,
        text_encoder: bool = False,
        vae: bool = False,
        controlnet: bool = False,
    ) -> str:
        if details:
            if text_encoder or vae:
                extra_module_world_size = self._get_extra_module_world_size()
                # Currently, only support tensor parallelism or data parallelism
                # for extra modules using pytorch native backend or pure pytorch
                # implementation. So we just hardcode the backend here.
                parallel_str = f"ParallelismConfig(backend={ParallelismBackend.NATIVE_PYTORCH}, "

                if text_encoder:
                    parallel_str += f"tp_size={extra_module_world_size}, "
                elif controlnet:
                    parallel_str += f"ulysses_size={extra_module_world_size}, "
                else:
                    parallel_str += f"dp_size={extra_module_world_size}, "
                parallel_str = parallel_str.rstrip(", ") + ")"
                return parallel_str

            parallel_str = f"ParallelismConfig(backend={self.backend}, "
            if self.ulysses_size is not None:
                parallel_str += f"ulysses_size={self.ulysses_size}, "
            if self.ring_size is not None:
                parallel_str += f"ring_size={self.ring_size}, "
            if self.tp_size is not None:
                parallel_str += f"tp_size={self.tp_size}, "
            parallel_str = parallel_str.rstrip(", ") + ")"
            return parallel_str
        else:
            parallel_str = ""
            if self.ulysses_size is not None:
                parallel_str += f"Ulysses{self.ulysses_size}"
            if self.ring_size is not None:
                parallel_str += f"Ring{self.ring_size}"
            if self.tp_size is not None:
                parallel_str += f"TP{self.tp_size}"
            if text_encoder:
                parallel_str += "_TEP"  # Text Encoder Parallelism
            if vae:
                parallel_str += "_VAEP"  # VAE Parallelism
            if controlnet:
                parallel_str += "_CNP"  # ControlNet Parallelism
            return parallel_str

    def _get_extra_module_world_size(self) -> Optional[int]:
        """Get the world size for extra parallel modules, e.g., text encoder and VAE."""
        # Maximize the parallel size for extra modules: max(tp_size, ulysses_size, ring_size)
        sizes = []
        if self.tp_size is not None and self.tp_size > 1:
            sizes.append(self.tp_size)
        if self.ulysses_size is not None and self.ulysses_size > 1:
            sizes.append(self.ulysses_size)
        if self.ring_size is not None and self.ring_size > 1:
            sizes.append(self.ring_size)
        if sizes:
            return max(sizes)
        return None

    @property
    def text_encoder_world_size(self) -> int:
        """Get the world size for text encoder parallelism."""
        world_size = self._get_extra_module_world_size()
        assert (
            world_size is None or world_size > 1
        ), "Text encoder world size must be None or greater than 1 for parallelism."
        return world_size

    @property
    def vae_world_size(self) -> int:
        """Get the world size for VAE parallelism."""
        world_size = self._get_extra_module_world_size()
        assert (
            world_size is None or world_size > 1
        ), "VAE world size must be None or greater than 1 for parallelism."
        return world_size
