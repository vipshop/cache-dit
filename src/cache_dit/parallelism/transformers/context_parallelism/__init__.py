import torch
import functools
import torch.distributed as dist
from typing import Optional

from diffusers.models.modeling_utils import ModelMixin
from cache_dit.parallelism.config import ParallelismConfig
from ....logger import init_logger

try:
    from ...attention import (
        _ExtendedContextParallelConfig,
        _enable_context_parallelism_ext,
        _maybe_register_custom_attn_backends,
        _is_diffusers_parallelism_available,
        enable_ulysses_anything,
        enable_ulysses_float8,
    )
    from .cp_plan_registers import ContextParallelismPlannerRegister
    from .cp_planners import _activate_cp_planners

    _maybe_register_custom_attn_backends()
    _activate_cp_planners()
except ImportError as e:
    raise ImportError(e)


logger = init_logger(__name__)


def maybe_enable_context_parallelism(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: Optional[ParallelismConfig] = None,
) -> torch.nn.Module:
    assert isinstance(transformer, (torch.nn.Module, ModelMixin)), (
        "transformer must be an instance of torch.nn.Module or ModelMixin, "
        f"but got {type(transformer)}"
    )
    if parallelism_config is None:
        return transformer

    assert isinstance(parallelism_config, ParallelismConfig), (
        "parallelism_config must be an instance of ParallelismConfig"
        f" but got {type(parallelism_config)}"
    )
    assert _is_diffusers_parallelism_available(), (
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source"
    )

    if parallelism_config.cp_enabled():
        # Prepare extra context parallelism config, e.g, convert_to_fp32,
        # rotate_method for ring attention.
        cp_config = _ExtendedContextParallelConfig(
            ulysses_degree=parallelism_config.ulysses_size,
            ring_degree=parallelism_config.ring_size,
            convert_to_fp32=parallelism_config.ring_convert_to_fp32,
            rotate_method=parallelism_config.ring_rotate_method,
        )
        if parallelism_config.hybrid_enabled():
            # In hybrid mode, we use the _cp_mesh from ParallelismConfig for
            # context parallelism.
            cp_config.setup(
                rank=parallelism_config._cp_rank,
                world_size=parallelism_config._cp_world_size,
                device=parallelism_config._device,
                mesh=parallelism_config._cp_mesh,
            )

        dynamic_sp_config = parallelism_config.dynamic_sp_config
        dynamic_sp_enabled = dynamic_sp_config is not None and dynamic_sp_config.enabled

        # Dynamic SP may switch to degrees that make CP split dimensions
        # temporarily not divisible by mesh.size(). Ulysses-anything replaces
        # strict equipartition with tensor_split/all-gather to support that.
        if parallelism_config.ulysses_anything or dynamic_sp_enabled:
            if dynamic_sp_enabled and not parallelism_config.ulysses_anything:
                logger.warning(
                    "dynamic_sp is enabled; auto enabling ulysses_anything to support "
                    "non-divisible per-step sharding."
                )
            enable_ulysses_anything()

        if parallelism_config.ulysses_float8:
            enable_ulysses_float8()

        # Prefer custom cp_plan if provided
        cp_plan = parallelism_config.cp_plan
        if cp_plan is not None:
            logger.info(f"Using custom context parallelism plan: {cp_plan}")
        else:
            # Try get context parallelism plan from register if not provided
            cp_plan = ContextParallelismPlannerRegister.get_planner(transformer)().apply(
                transformer=transformer, parallelism_config=parallelism_config
            )

        _enable_context_parallelism_ext(transformer, config=cp_config, cp_plan=cp_plan)
        _maybe_patch_native_parallel_config(transformer)

        if dynamic_sp_config is not None and dynamic_sp_config.enabled:
            from ...dynamic_sp import DynamicSPManager

            rank = getattr(cp_config, "_rank", None)
            world_size = getattr(cp_config, "_world_size", None)
            if rank is None or world_size is None:
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            device_type = torch._C._get_accelerator().type
            manager = DynamicSPManager(
                config=dynamic_sp_config,
                cp_config=cp_config,
                rank=rank,
                world_size=world_size,
                device_type=device_type,
            )
            transformer._dynamic_sp_manager = manager
            _wrap_forward_with_dynamic_sp(transformer, manager)

    return transformer


def _wrap_forward_with_dynamic_sp(transformer: torch.nn.Module, manager) -> torch.nn.Module:
    original_forward = transformer.forward
    _hf_hook = getattr(transformer, "_hf_hook", None)
    # Align with cache adapter wrapping behavior:
    # when `_hf_hook` is present, `forward` may already be a hook wrapper.
    # Calling `_hf_hook.pre/post` around that wrapped `forward` would apply
    # hooks twice. Prefer `_old_forward` when available.
    if _hf_hook is not None and hasattr(transformer, "_old_forward"):
        original_forward = transformer._old_forward

    def new_forward(self, *args, **kwargs):
        step = manager.step
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and len(args) > 0:
            hidden_states = args[0]

        output = None
        if manager.is_active(step):
            manager.apply_config(step)
            output = original_forward(*args, **kwargs)

        output = manager.sync_output(output=output, hidden_states=hidden_states, step=step)
        manager.advance_step()
        return output

    def new_forward_with_hf_hook(self, *args, **kwargs):
        if _hf_hook is not None and hasattr(_hf_hook, "pre_forward"):
            args, kwargs = _hf_hook.pre_forward(self, *args, **kwargs)

        outputs = new_forward(self, *args, **kwargs)

        if _hf_hook is not None and hasattr(_hf_hook, "post_forward"):
            outputs = _hf_hook.post_forward(self, outputs)
        return outputs

    transformer.forward = functools.update_wrapper(
        functools.partial(new_forward_with_hf_hook, transformer),
        new_forward_with_hf_hook,
    )
    transformer._dynamic_sp_original_forward = original_forward
    return transformer


def _maybe_patch_native_parallel_config(
    transformer: torch.nn.Module,
    **kwargs,
) -> torch.nn.Module:

    cls_name = transformer.__class__.__name__
    if not cls_name.startswith("Nunchaku"):
        return transformer

    try:
        from nunchaku.models.transformers.transformer_flux_v2 import (
            NunchakuFluxTransformer2DModelV2,
            NunchakuFluxAttention,
            NunchakuFluxFA2Processor,
        )
        from nunchaku.models.transformers.transformer_qwenimage import (
            NunchakuQwenAttention,
            NunchakuQwenImageNaiveFA2Processor,
            NunchakuQwenImageTransformer2DModel,
        )
        from nunchaku.models.transformers.transformer_zimage import (
            NunchakuZImageTransformer2DModel,
            NunchakuZSingleStreamAttnProcessor,
            NunchakuZImageAttention,
        )
    except ImportError:
        raise ImportError(
            "NunchakuZImageTransformer2DModel, NunchakuFluxTransformer2DModelV2 and "
            "NunchakuQwenImageTransformer2DModel requires the 'nunchaku' package. "
            "Please install nunchaku>=1.10 before using the context parallelism for "
            "nunchaku 4-bits models."
        )

    assert isinstance(
        transformer,
        (
            NunchakuFluxTransformer2DModelV2,
            NunchakuQwenImageTransformer2DModel,
            NunchakuZImageTransformer2DModel,
        ),
    )
    config = getattr(transformer, "_parallel_config", None)
    if config is None:
        raise logger.warning(
            f"The transformer {cls_name} does not have _parallel_config attribute. "
            "Skipping patching native parallel config."
        )

    attention_classes = (
        NunchakuFluxAttention,
        NunchakuFluxFA2Processor,
        NunchakuQwenAttention,
        NunchakuQwenImageNaiveFA2Processor,
        NunchakuZImageAttention,
        NunchakuZSingleStreamAttnProcessor,
    )
    for module in transformer.modules():
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "_parallel_config"):
            continue
        if getattr(processor, "_parallel_config", None) is not None:
            logger.warning(
                f"The attention processor {processor.__class__.__name__} already has "
                "_parallel_config attribute set. Skipping patching native parallel config."
            )
            continue
        processor._parallel_config = config

    return transformer
