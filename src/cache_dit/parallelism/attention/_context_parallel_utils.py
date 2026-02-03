import torch
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union
from diffusers import ModelMixin
from diffusers import ContextParallelConfig, ParallelConfig
from diffusers.models._modeling_parallel import ContextParallelModelPlan
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _ExtendedContextParallelConfig(ContextParallelConfig):
    rotate_method: Literal["allgather", "alltoall", "p2p"] = "allgather"
    extra_kwargs: Dict[str, Any] = None  # For future extensions

    def __post_init__(self):
        # Override the __post_init__ method to allow the extended features
        # in cache-dit to work properly.
        if self.ring_degree is None:
            self.ring_degree = 1
        if self.ulysses_degree is None:
            self.ulysses_degree = 1

        if self.ring_degree == 1 and self.ulysses_degree == 1:
            raise ValueError(
                "Either ring_degree or ulysses_degree must be greater than 1 in order "
                "to use context parallel inference"
            )
        if self.ring_degree < 1 or self.ulysses_degree < 1:
            raise ValueError(
                "`ring_degree` and `ulysses_degree` must be greater than or equal to 1."
            )
        if self.rotate_method not in ["allgather", "p2p"]:
            raise NotImplementedError(
                "Only the 'allgather' and 'p2p' rotate methods are supported for now, "
                f"but got {self.rotate_method}."
            )


# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/modeling_utils.py#L1510
def _enable_parallelism_ext(
    model: ModelMixin | torch.nn.Module,  # e.g Transformer
    *,
    config: Union[ParallelConfig, ContextParallelConfig],
    cp_plan: Optional[Dict[str, ContextParallelModelPlan]] = None,
    attn_classes_extra: Optional[tuple] = None,
):
    logger.info("Dispatch parallelism using _enable_parallelism_ext in cache-dit.")

    if not torch.distributed.is_available() and not torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed must be available and initialized before calling `enable_parallelism`."
        )
    from diffusers.hooks.context_parallel import apply_context_parallel
    from diffusers.models.attention import AttentionModuleMixin
    from diffusers.models.attention_dispatch import (
        AttentionBackendName,
        _AttentionBackendRegistry,
    )
    from diffusers.models.attention_processor import Attention, MochiAttention

    if isinstance(config, ContextParallelConfig):
        config = ParallelConfig(context_parallel_config=config)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device_type = torch._C._get_accelerator().type
    device_module = torch.get_device_module(device_type)
    device = torch.device(device_type, rank % device_module.device_count())

    # TODO: Support non-diffusers models in the future, e.g., HuggingFace
    # Transformers or custom models.
    attention_classes = (Attention, MochiAttention, AttentionModuleMixin)
    if attn_classes_extra is not None:
        attention_classes += attn_classes_extra

    if config.context_parallel_config is not None:
        for module in model.modules():
            if not isinstance(module, attention_classes):
                continue

            processor = module.processor
            if processor is None or not hasattr(processor, "_attention_backend"):
                # Auto attach the _attention_backend attribute if not present
                processor._attention_backend = None

            attention_backend = processor._attention_backend
            if attention_backend is None:
                attention_backend, _ = _AttentionBackendRegistry.get_active_backend()
            else:
                attention_backend = AttentionBackendName(attention_backend)

            if not _AttentionBackendRegistry._is_context_parallel_available(attention_backend):
                compatible_backends = sorted(_AttentionBackendRegistry._supports_context_parallel)
                raise ValueError(
                    f"Context parallelism is enabled but the attention processor '{processor.__class__.__name__}' "
                    f"is using backend '{attention_backend.value}' which does not support context parallelism. "
                    f"Please set a compatible attention backend: {compatible_backends} using `set_attn_backend()` before "
                    f"calling `_enable_parallelism_ext()`."
                )

            # All modules use the same attention processor and backend. We don't need to
            # iterate over all modules after checking the first processor
            break

    mesh = None
    if config.context_parallel_config is not None:
        cp_config = config.context_parallel_config

        # NOTE(DefTruth): Allow user to pass in a custom mesh
        if cp_config._mesh is None:
            mesh = torch.distributed.device_mesh.init_device_mesh(
                device_type=device_type,
                mesh_shape=cp_config.mesh_shape,
                mesh_dim_names=cp_config.mesh_dim_names,
            )
            config.setup(rank, world_size, device, mesh=mesh)

    # Will auto attach the _parallel_config attribute to the model
    # if not present implicitly by python's dynamic attr setting.
    model._parallel_config = config  # type: ignore[attr-defined]

    for module in model.modules():
        if not isinstance(module, attention_classes):
            continue

        processor = module.processor
        if processor is None or not hasattr(processor, "_parallel_config"):
            # Auto attach the _parallel_config attribute if not present
            processor._parallel_config = None
        processor._parallel_config = config

    if config.context_parallel_config is not None:
        if cp_plan is None and model._cp_plan is None:
            raise ValueError(
                "`cp_plan` must be provided either as an argument or set in the model's `_cp_plan` attribute."
            )
        cp_plan = cp_plan if cp_plan is not None else model._cp_plan
        # NOTE: apply_context_parallel only requires the model is an instance of
        # torch.nn.Module, thus, it open the possibility to support non-diffusers
        # models in the future.
        apply_context_parallel(model, config.context_parallel_config, cp_plan)
