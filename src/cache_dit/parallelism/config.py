import torch
import dataclasses
from typing import Optional, Dict, Any
import torch.distributed as dist
from .backend import ParallelismBackend
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class ParallelismConfig:
    # Parallelism backend, defaults to AUTO. We will auto select the backend
    # based on the parallelism configuration.
    backend: ParallelismBackend = ParallelismBackend.AUTO
    # Context parallelism config
    # ulysses_size (`int`, *optional*):
    #   The degree of ulysses parallelism.
    ulysses_size: int = None
    # ring_size (`int`, *optional*):
    #   The degree of ring parallelism.
    ring_size: int = None
    # Tensor parallelism config
    # tp_size (`int`, *optional*):
    #   The degree of tensor parallelism.
    tp_size: int = None
    # parallel_kwargs (`dict`, *optional*):
    #   Additional kwargs for parallelism backends. For example, for
    #   NATIVE_DIFFUSER backend, it can include:
    #   `cp_plan`: The custom context parallelism plan pass by user.
    #   `attention_backend`: str, The attention backend for parallel attention,
    #       e.g, 'native', 'flash', 'sage', etc.
    #   `experimental_ulysses_anything: bool, Whether to enable the ulysses
    #       anything attention to support arbitrary sequence length and
    #       arbitrary number of heads.
    #   `experimental_ulysses_async: bool, Whether to enable the ulysses async
    #       attention to overlap communication and computation.
    #   `experimental_ulysses_float8: bool, Whether to enable the ulysses float8
    #       attention to use fp8 for faster communication.
    #   `ring_rotate_method`: str, The ring rotate method, default is `p2p`:
    #       'p2p': Use batch_isend_irecv ops to rotate the key and value tensors.
    #            This method is more efficient due to th better overlap of communication
    #            and computation (default)
    #       'allgather': Use allgather to gather the key and value tensors.
    #   `ring_convert_to_fp32`: bool, Whether to convert the value output and lse
    #       of ring attention to fp32. Default to True to avoid numerical issues.
    parallel_kwargs: Optional[Dict[str, Any]] = dataclasses.field(default_factory=dict)

    # Flags to indicate whether the model has extra modules that need parallelism
    _has_text_encoder: bool = False
    _has_auto_encoder: bool = False
    _has_controlnet: bool = False

    # Meshes for hybrid parallelism: CP/SP + TP
    _mesh: Optional[dist.device_mesh.DeviceMesh] = None
    _cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None
    _tp_mesh: Optional[dist.device_mesh.DeviceMesh] = None
    _flat_mesh: Optional[dist.device_mesh.DeviceMesh] = None
    _flat_cp_mesh: Optional[dist.device_mesh.DeviceMesh] = None
    _flat_tp_mesh: Optional[dist.device_mesh.DeviceMesh] = None
    _rank: Optional[int] = None
    _cp_rank: Optional[int] = None
    _tp_rank: Optional[int] = None
    _world_size: Optional[int] = None
    _cp_world_size: Optional[int] = None
    _tp_world_size: Optional[int] = None
    _device: Optional[torch.device] = None
    _device_type: Optional[str] = None
    _device_module: Optional[Any] = None

    def __post_init__(self):
        assert ParallelismBackend.is_supported(self.backend), (
            f"Parallel backend {self.backend} is not supported. "
            f"Please make sure the required packages are installed."
        )
        if self.backend == ParallelismBackend.AUTO:
            # Auto select the backend based on the parallelism configuration
            if self.hybrid_enabled():
                self.backend = ParallelismBackend.NATIVE_HYBRID
            elif self.cp_enabled() or self.usp_enabled():
                self.backend = ParallelismBackend.NATIVE_DIFFUSER
            elif self.tp_enabled():
                self.backend = ParallelismBackend.NATIVE_PYTORCH
            else:
                self.backend = ParallelismBackend.NONE
            logger.info(f"Auto selected parallelism backend for transformer: {self.backend}")

        world_size = self._get_world_size()
        if self.hybrid_enabled():
            assert world_size >= 4, (
                "Hybrid Ulysses + Ring + TP parallelism requires at least 4 processes. "
                f"Got {world_size} processes."
            )
            if self.usp_enabled():
                assert world_size >= 8, (
                    "Hybrid Ulysses + Ring + TP parallelism requires at least 8 processes. "
                    f"Got {world_size} processes."
                )
        if self.usp_enabled():
            assert world_size >= 4, (
                "Ulysses + Ring parallelism requires at least 4 processes. "
                f"Got {world_size} processes."
            )

        # Validate the parallelism configuration and auto adjust the backend if needed
        if self.hybrid_enabled():
            assert (
                self.backend == ParallelismBackend.NATIVE_HYBRID
            ), "Hybrid parallelism requires the backend to be NATIVE_HYBRID."
        elif self.cp_enabled() or self.usp_enabled():
            assert (
                self.backend == ParallelismBackend.NATIVE_DIFFUSER
            ), "Context parallelism requires the backend to be NATIVE_DIFFUSER."
        elif self.tp_enabled():
            assert (
                self.backend == ParallelismBackend.NATIVE_PYTORCH
            ), "Tensor parallelism requires the backend to be NATIVE_PYTORCH."
        else:
            raise ValueError(
                "No parallelism is enabled. Please set ulysses_size, ring_size, or tp_size "
                "to enable parallelism."
            )

        if self.hybrid_enabled():
            try:
                self._maybe_init_hybrid_meshes()
            except Exception as e:
                # Required: https://github.com/pytorch/pytorch/pull/158899/changes#diff-dbbed99b01763453143e50565b636cb37f8f693aefaf18b57d621781114ed1b7
                # Related issue: https://github.com/pytorch/pytorch/issues/159013
                # The hybrid 3D parallelism scheme in cache-dit is:
                # [ring, ulysses, tp] -> slice to [ring, ulysses] + [tp] -> pass [ring, ulysses] to diffusers CP backend,
                # then diffusers CP backend will slice [ring] and [ulysses] from the submesh [ring, ulysses]
                # (namely, [ring, ulysses][ring] -> ring mesh, [ring, ulysses][ulysses] -> ulysses mesh) for
                # ring and ulysses parallelism. However, in older PyTorch versions, creating a submesh from
                # a submesh is not supported, which will raise the error "Cannot create a submesh from a submesh".
                # Here we catch this specific error and provide a more user-friendly message.
                err_msg = str(e)
                hit_msg = "Cannot create a submesh from a submesh"
                if hit_msg in err_msg:
                    err_msg += (
                        "\nThis is likely due to using an older version of PyTorch that does not "
                        "support creating submeshes from submeshes. Please upgrade to PyTorch "
                        "2.10.0 or later."
                    )
                raise RuntimeError(err_msg) from e

    def _maybe_init_hybrid_meshes(self):
        if self._mesh is not None or not self.hybrid_enabled():
            return  # already initialized or not hybrid enabled
        self._rank = dist.get_rank()
        self._world_size = dist.get_world_size()
        self._device_type = torch._C._get_accelerator().type
        self._device_module = torch.get_device_module(self._device_type)
        self._device = torch.device(
            self._device_type,
            self._rank % self._device_module.device_count(),
        )
        # 3d mesh (ring, ulysses, tp) -> 2d cp mesh (ring * ulysses, ) + 1d tp mesh
        ring_size = self.ring_size if self.ring_size is not None else 1
        ulysses_size = self.ulysses_size if self.ulysses_size is not None else 1
        tp_size = self.tp_size if self.tp_size is not None else 1

        self._mesh = dist.device_mesh.init_device_mesh(
            device_type=self._device_type,
            mesh_shape=(ring_size, ulysses_size, tp_size),
            mesh_dim_names=("ring", "ulysses", "tp"),
        )

        # Slice cp_mesh and tp_mesh and infer special ranks and world sizes
        self._cp_mesh = self._mesh["ring", "ulysses"]
        self._tp_mesh = self._mesh["tp"]
        self._flat_mesh = self._mesh._flatten()
        self._flat_cp_mesh = self._cp_mesh._flatten()
        try:
            self._flat_tp_mesh = self._tp_mesh._flatten()
        except Exception as e:
            # Workaround for error: SGLang Diffusion + diffusers BE + cache-dit hybrid parallelism
            # + PyTorch <= 2.9, which may not support creating flat mesh from last dim of submesh.
            # RuntimeError: ("tp already exists for submesh of the DeviceMesh((ring=1, ulysses=2, tp=2)
            # NOTE: flat_tp_mesh is only used for tensor parallelism, which is 1D and does not require
            # slicing. So we can just use the original tp_mesh as the flat_tp_mesh without flattening.
            if "tp already exists for submesh" in str(e):
                self._flat_tp_mesh = self._tp_mesh
            else:
                raise e
        self._rank = self._flat_mesh.get_local_rank()
        self._cp_rank = self._flat_cp_mesh.get_local_rank()
        self._tp_rank = self._flat_tp_mesh.get_local_rank()
        self._world_size = self._flat_mesh.size()
        self._cp_world_size = self._flat_cp_mesh.size()
        self._tp_world_size = self._flat_tp_mesh.size()

    def enabled(self) -> bool:
        return (
            (self.ulysses_size is not None and self.ulysses_size > 1)
            or (self.ring_size is not None and self.ring_size > 1)
            or (self.tp_size is not None and self.tp_size > 1)
        )

    def cp_enabled(self) -> bool:
        return (self.ulysses_size is not None and self.ulysses_size > 1) or (
            self.ring_size is not None and self.ring_size > 1
        )

    def tp_enabled(self) -> bool:
        return self.tp_size is not None and self.tp_size > 1

    def usp_enabled(self) -> bool:
        return (
            self.ulysses_size is not None
            and self.ulysses_size > 1
            and self.ring_size is not None
            and self.ring_size > 1
        )

    def hybrid_enabled(self) -> bool:
        return self.cp_enabled() and self.tp_enabled()

    def strify(
        self,
        details: bool = False,
        text_encoder: bool = False,
        vae: bool = False,
        controlnet: bool = False,
    ) -> str:
        if details:
            if text_encoder or vae:
                extra_module_world_size = self._get_world_size()
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
                parallel_str += f"Ulysses{self.ulysses_size}_"
            if self.ring_size is not None:
                parallel_str += f"Ring{self.ring_size}_"
            if self.tp_size is not None:
                parallel_str += f"TP{self.tp_size}_"
            if text_encoder or self._has_text_encoder:
                parallel_str += "TEP_"  # Text Encoder Parallelism
            if vae or self._has_auto_encoder:
                parallel_str += "VAEP_"  # VAE Parallelism
            if controlnet or self._has_controlnet:
                parallel_str += "CNP"  # ControlNet Parallelism
            parallel_str = parallel_str.rstrip("_")
            return parallel_str

    def _get_world_size(self) -> Optional[int]:
        """Get the world size for extra parallel modules, e.g., text encoder and VAE."""
        # Maximize the parallel size for extra modules
        sizes = []
        ring_size = self.ring_size if self.ring_size is not None else 1
        ulysses_size = self.ulysses_size if self.ulysses_size is not None else 1
        tp_size = self.tp_size if self.tp_size is not None else 1

        if self.hybrid_enabled():
            sizes.append(ulysses_size * ring_size * tp_size)
        elif self.usp_enabled():
            sizes.append(ulysses_size * ring_size)
        elif self.cp_enabled():
            sizes.append(max(ulysses_size, ring_size))
        elif self.tp_enabled():
            sizes.append(tp_size)

        if sizes:
            return max(sizes)
        return 1

    @property
    def text_encoder_world_size(self) -> int:
        """Get the world size for text encoder parallelism."""
        world_size = self._get_world_size()
        self._has_text_encoder = True
        return world_size

    @property
    def auto_encoder_world_size(self) -> int:
        """Get the world size for VAE parallelism."""
        world_size = self._get_world_size()
        self._has_auto_encoder = True
        return world_size

    @property
    def vae_world_size(self) -> int:  # alias of auto_encoder_world_size
        return self.vae_world_size

    @property
    def controlnet_world_size(self) -> int:
        """Get the world size for ControlNet parallelism."""
        world_size = self._get_world_size()
        self._has_controlnet = True
        return world_size
