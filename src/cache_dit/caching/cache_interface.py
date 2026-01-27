import copy
import torch
from typing import Any, Tuple, List, Union, Optional
from diffusers import DiffusionPipeline, ModelMixin
from .cache_types import CacheType
from .block_adapters import BlockAdapter
from .block_adapters import BlockAdapterRegister
from .cache_adapters import CachedAdapter
from .cache_contexts import BasicCacheConfig
from .cache_contexts import DBCacheConfig
from .cache_contexts import DBPruneConfig
from .cache_contexts import CalibratorConfig
from .params_modifier import ParamsModifier
from ..parallelism import ParallelismConfig
from ..parallelism import enable_parallelism

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def enable_cache(
    # DiffusionPipeline or BlockAdapter
    pipe_or_adapter: Union[
        DiffusionPipeline,
        BlockAdapter,
        # Transformer-only
        torch.nn.Module,
        ModelMixin,
    ],
    # BasicCacheConfig, DBCacheConfig, DBPruneConfig, etc.
    cache_config: Optional[
        Union[
            BasicCacheConfig,
            DBCacheConfig,
            DBPruneConfig,
        ]
    ] = None,
    # Calibrator config: TaylorSeerCalibratorConfig, etc.
    calibrator_config: Optional[CalibratorConfig] = None,
    # Modify cache context params for specific blocks.
    params_modifiers: Optional[
        Union[
            ParamsModifier,
            List[ParamsModifier],
            List[List[ParamsModifier]],
        ]
    ] = None,
    # Config for Parallelism
    parallelism_config: Optional[ParallelismConfig] = None,
    # Allow set custom attention backend for non-parallelism case
    attention_backend: Optional[str] = None,
    # Other cache context kwargs: Deprecated cache kwargs
    **kwargs,
) -> Union[
    DiffusionPipeline,
    # Transformer-only
    torch.nn.Module,
    ModelMixin,
    BlockAdapter,
]:
    r"""
    The `enable_cache` function serves as a unified caching interface designed to optimize the performance
    of diffusion transformer models by implementing an intelligent caching mechanism known as `DBCache`.
    This API is engineered to be compatible with nearly `all` diffusion transformer architectures that
    feature transformer blocks adhering to standard input-output patterns, eliminating the need for
    architecture-specific modifications.

    By strategically caching intermediate outputs of transformer blocks during the diffusion process,
    `DBCache` significantly reduces redundant computations without compromising generation quality.
    The caching mechanism works by tracking residual differences between consecutive steps, allowing
    the model to reuse previously computed features when these differences fall below a configurable
    threshold. This approach maintains a balance between computational efficiency and output precision.

    The default configuration (`F8B0, 8 warmup steps, unlimited cached steps`) is carefully tuned to
    provide an optimal tradeoff for most common use cases. The "F8B0" configuration indicates that
    the first 8 transformer blocks are used to compute stable feature differences, while no final
    blocks are employed for additional fusion. The warmup phase ensures the model establishes
    sufficient feature representation before caching begins, preventing potential degradation of
    output quality.

    This function seamlessly integrates with both standard diffusion pipelines and custom block
    adapters, making it versatile for various deployment scenarios—from research prototyping to
    production environments where inference speed is critical. By abstracting the complexity of
    caching logic behind a simple interface, it enables developers to enhance model performance
    with minimal code changes.

    Args:
        pipe_or_adapter (`DiffusionPipeline`, `BlockAdapter` or `Transformer`, *required*):
            The standard Diffusion Pipeline or custom BlockAdapter (from cache-dit or user-defined).
            For example: cache_dit.enable_cache(FluxPipeline(...)).

        cache_config (`BasicCacheConfig`, *required*, defaults to BasicCacheConfig()):
            Basic DBCache config for cache context, defaults to BasicCacheConfig(). The configurable params listed belows:
                Fn_compute_blocks: (`int`, *required*, defaults to 8):
                    Specifies that `DBCache` uses the**first n**Transformer blocks to fit the information at time step t,
                    enabling the calculation of a more stable L1 difference and delivering more accurate information
                    to subsequent blocks. Please check https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md
                    for more details of DBCache.
                Bn_compute_blocks: (`int`, *required*, defaults to 0):
                    Further fuses approximate information in the **last n** Transformer blocks to enhance
                    prediction accuracy. These blocks act as an auto-scaler for approximate hidden states
                    that use residual cache.
                residual_diff_threshold (`float`, *required*, defaults to 0.08):
                    the value of residual diff threshold, a higher value leads to faster performance at the
                    cost of lower precision.
                max_accumulated_residual_diff_threshold (`float`, *optional*, defaults to None):
                    The maximum accumulated relative l1 diff threshold for Cache. If set, when the
                    accumulated relative l1 diff exceeds this threshold, the caching strategy will be
                    disabled for current step. This is useful for some cases where the input condition
                    changes significantly in a single step. Default None means this feature is disabled.
                max_warmup_steps (`int`, *required*, defaults to 8):
                    DBCache does not apply the caching strategy when the number of running steps is less than
                    or equal to this value, ensuring the model sufficiently learns basic features during warmup.
                warmup_interval (`int`, *required*, defaults to 1):
                    Skip interval in warmup steps, e.g., when warmup_interval is 2, only 0, 2, 4, ... steps
                    in warmup steps will be computed, others will use dynamic cache.
                max_cached_steps (`int`, *required*, defaults to -1):
                    DBCache disables the caching strategy when the previous cached steps exceed this value to
                    prevent precision degradation.
                max_continuous_cached_steps (`int`, *required*, defaults to -1):
                    DBCache disables the caching strategy when the previous continous cached steps exceed this value to
                    prevent precision degradation.
                enable_separate_cfg (`bool`, *required*,  defaults to None):
                    Whether to do separate cfg or not, such as Wan 2.1, Qwen-Image. For model that fused CFG
                    and non-CFG into single forward step, should set enable_separate_cfg as False, for example:
                    CogVideoX, HunyuanVideo, Mochi, etc.
                cfg_compute_first (`bool`, *required*,  defaults to False):
                    Whether to compute cfg forward first, default is False, meaning:
                    0, 2, 4, ..., -> non-CFG step;
                    1, 3, 5, ... -> CFG step.
                cfg_diff_compute_separate (`bool`, *required*,  defaults to True):
                    Whether to compute separate difference values for CFG and non-CFG steps, default is True.
                    If False, we will use the computed difference from the current non-CFG transformer step
                    for the current CFG step.
                num_inference_steps (`int`, *optional*, defaults to None):
                    num_inference_steps for DiffusionPipeline, used to adjust some internal settings
                    for better caching performance. For example, we will refresh the cache once the
                    executed steps exceed num_inference_steps if num_inference_steps is provided.
                steps_computation_mask (`List[int]`, *optional*, defaults to None):
                    This param introduce LeMiCa/EasyCache style compute mask for steps. It is a list
                    of length num_inference_steps indicating whether to compute each step or not.
                    1 means must compute, 0 means use dynamic/static cache. If provided, will override
                    other settings to decide whether to compute each step.
                steps_computation_policy (`str`, *optional*, defaults to "dynamic"):
                    The computation policy for steps when using steps_computation_mask. It can be
                    "dynamic" or "static". "dynamic" means using dynamic cache for steps marked as 0
                    in steps_computation_mask, while "static" means using static cache for those steps.

        calibrator_config (`CalibratorConfig`, *optional*, defaults to None):
            Config for calibrator. If calibrator_config is not None, it means the user wants to use DBCache
            with a specific calibrator, such as taylorseer, foca, and so on.

        params_modifiers ('ParamsModifier', *optional*, defaults to None):
            Modify cache context params for specific blocks. The configurable params listed belows:
                cache_config (`BasicCacheConfig`, *required*, defaults to BasicCacheConfig()):
                    The same as 'cache_config' param in cache_dit.enable_cache() interface.
                calibrator_config (`CalibratorConfig`, *optional*, defaults to None):
                    The same as 'calibrator_config' param in cache_dit.enable_cache() interface.
                **kwargs: (`dict`, *optional*, defaults to {}):
                    The same as 'kwargs' param in cache_dit.enable_cache() interface.

        parallelism_config (`ParallelismConfig`, *optional*, defaults to None):
            Config for Parallelism. If parallelism_config is not None, it means the user wants to enable
            parallelism for cache-dit. Please check https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/parallel_config.py
            for more details of ParallelismConfig.
                backend: (`ParallelismBackend`, *required*, defaults to "ParallelismBackend.NATIVE_DIFFUSER"):
                    Parallelism backend, currently only NATIVE_DIFFUSER and NVTIVE_PYTORCH are supported.
                    For context parallelism, only NATIVE_DIFFUSER backend is supported, for tensor parallelism,
                    only NATIVE_PYTORCH backend is supported.
                ulysses_size: (`int`, *optional*, defaults to None):
                    The size of Ulysses cluster. If ulysses_size is not None, enable Ulysses style parallelism.
                    This setting is only valid when backend is NATIVE_DIFFUSER.
                ring_size: (`int`, *optional*, defaults to None):
                    The size of ring for ring parallelism. If ring_size is not None, enable ring attention.
                    This setting is only valid when backend is NATIVE_DIFFUSER.
                tp_size: (`int`, *optional*, defaults to None):
                    The size of tensor parallelism. If tp_size is not None, enable tensor parallelism.
                    This setting is only valid when backend is NATIVE_PYTORCH.
                parallel_kwargs: (`dict`, *optional*, defaults to {}):
                    Additional kwargs for parallelism backends. For example, for NATIVE_DIFFUSER backend,
                    it can include `cp_plan` and `attention_backend` arguments for `Context Parallelism`.
                    Additional kwargs for parallelism backends. For example, for NATIVE_DIFFUSER backend,
                    it can include:
                        cp_plan: The custom context parallelism plan pass by user.
                        attention_backend: str, The attention backend for parallel attention, e.g, 'native', 'flash', 'sage', etc.
                        experimental_ulysses_anything: bool, Whether to enable the ulysses anything attention to support arbitrary sequence length and arbitrary number of heads.
                        experimental_ulysses_async: bool, Whether to enable the ulysses async attention to overlap communication and computation.
                        experimental_ulysses_float8: bool, Whether to enable the ulysses float8 attention to use fp8 for faster communication.
                        ring_rotate_method: str, The ring rotate method, default is `p2p`:
                            `p2p`: Use batch_isend_irecv ops to rotate the key and value tensors. This method is more efficient due to th better overlap of communication and computation (default).
                            `allgather`: Use allgather to gather the key and value tensors.
                        ring_convert_to_fp32: bool, Whether to convert the value output and lse of ring attention to fp32. Default to True to avoid numerical issues.

        attention_backend (`str`, *optional*, defaults to None):
            Custom attention backend in cache-dit for non-parallelism case. If attention_backend is
            not None, set the attention backend for the transformer module. Supported backends include:
            "native", "_sdpa_cudnn", "sage", "flash", "flash", "_native_npu", etc. Prefer attention_backend
            in parallelism_config when both are provided.

        kwargs (`dict`, *optional*, defaults to {})
            Other cache context kwargs, please check https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/cache_contexts/cache_context.py
            for more details.

    Examples:
    ```py
    >>> import cache_dit
    >>> from diffusers import DiffusionPipeline
    >>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image") # Can be any diffusion pipeline
    >>> cache_dit.enable_cache(pipe) # One-line code with default cache options.
    >>> output = pipe(...) # Just call the pipe as normal.
    >>> stats = cache_dit.summary(pipe) # Then, get the summary of cache acceleration stats.
    >>> cache_dit.disable_cache(pipe) # Disable cache and run original pipe.
    """
    # Precheck for compatibility of different configurations
    if cache_config is None:
        if parallelism_config is None:
            # Set default cache config only when parallelism is not enabled
            logger.info("cache_config is None, using default DBCacheConfig")
            cache_config = DBCacheConfig()
        else:
            # Allow empty cache_config when parallelism is enabled
            logger.warning(
                "Parallelism is enabled and cache_config is None. Please manually "
                "set cache_config to avoid potential compatibility issues. "
                "Otherwise, cache will not be enabled."
            )

    # Collect cache context kwargs
    context_kwargs = {}
    if (cache_type := context_kwargs.get("cache_type", None)) is not None:
        if cache_type == CacheType.NONE:
            return pipe_or_adapter

    # NOTE: Deprecated cache config params. These parameters are now retained
    # for backward compatibility but will be removed in the future.
    deprecated_kwargs = {
        "Fn_compute_blocks": kwargs.get("Fn_compute_blocks", None),
        "Bn_compute_blocks": kwargs.get("Bn_compute_blocks", None),
        "max_warmup_steps": kwargs.get("max_warmup_steps", None),
        "max_cached_steps": kwargs.get("max_cached_steps", None),
        "max_continuous_cached_steps": kwargs.get("max_continuous_cached_steps", None),
        "residual_diff_threshold": kwargs.get("residual_diff_threshold", None),
        "enable_separate_cfg": kwargs.get("enable_separate_cfg", None),
        "cfg_compute_first": kwargs.get("cfg_compute_first", None),
        "cfg_diff_compute_separate": kwargs.get("cfg_diff_compute_separate", None),
    }

    deprecated_kwargs = {k: v for k, v in deprecated_kwargs.items() if v is not None}

    if deprecated_kwargs:
        logger.warning(
            "Manually settup DBCache context without BasicCacheConfig is "
            "deprecated and will be removed in the future, please use "
            "`cache_config` parameter instead!"
        )
        if cache_config is not None:
            cache_config.update(**deprecated_kwargs)
        else:
            cache_config = BasicCacheConfig(**deprecated_kwargs)

    if cache_config is not None:
        context_kwargs["cache_config"] = cache_config

    # NOTE: Deprecated taylorseer params. These parameters are now retained
    # for backward compatibility but will be removed in the future.
    if cache_config is not None and (
        kwargs.get("enable_taylorseer", None) is not None
        or kwargs.get("enable_encoder_taylorseer", None) is not None
    ):
        logger.warning(
            "Manually settup TaylorSeer calibrator without TaylorSeerCalibratorConfig is "
            "deprecated and will be removed in the future, please use "
            "`calibrator_config` parameter instead!"
        )
        from .cache_contexts.calibrators import (
            TaylorSeerCalibratorConfig,
        )

        calibrator_config = TaylorSeerCalibratorConfig(
            enable_calibrator=kwargs.get("enable_taylorseer"),
            enable_encoder_calibrator=kwargs.get("enable_encoder_taylorseer"),
            calibrator_cache_type=kwargs.get("taylorseer_cache_type", "residual"),
            taylorseer_order=kwargs.get("taylorseer_order", 1),
        )

    if calibrator_config is not None:
        context_kwargs["calibrator_config"] = calibrator_config

    if params_modifiers is not None:
        context_kwargs["params_modifiers"] = params_modifiers

    if cache_config is not None:
        if isinstance(
            pipe_or_adapter,
            (DiffusionPipeline, BlockAdapter, torch.nn.Module, ModelMixin),
        ):
            pipe_or_adapter = CachedAdapter.apply(
                pipe_or_adapter,
                **context_kwargs,
            )
        else:
            raise ValueError(
                f"type: {type(pipe_or_adapter)} is not valid, "
                "Please pass DiffusionPipeline or BlockAdapter"
                "for the 1's position param: pipe_or_adapter"
            )
    else:
        logger.warning(
            "cache_config is None, skip enabling cache for "
            f"{pipe_or_adapter.__class__.__name__}."
        )

    # Set custom attention backend for non-parallelism case
    if attention_backend is not None:
        if parallelism_config is not None:
            if "attention_backend" in parallelism_config.parallel_kwargs:
                logger.warning(
                    "Both attention_backend in parallelism_config and "
                    "attention_backend param are provided, prefer using "
                    "attention_backend in parallelism_config."
                )
                attention_backend = None  # Prefer parallelism_config setting
            else:
                logger.info(
                    "Setting attention_backend from attention_backend "
                    "param to parallelism_config."
                )
                parallelism_config.parallel_kwargs["attention_backend"] = attention_backend
        else:
            set_attn_backend(pipe_or_adapter, attention_backend)

    # NOTE: Users should always enable parallelism after applying
    # cache to avoid hooks conflict.
    if parallelism_config is not None:
        assert isinstance(
            parallelism_config, ParallelismConfig
        ), "parallelism_config should be of type ParallelismConfig."

        # Prefer custom has_controlnet flag from users if provided, otherwise,
        # we will automatically check whether the pipeline has controlnet.
        if "has_controlnet" not in parallelism_config.parallel_kwargs:
            # This flag is used to decide whether to use the special parallelism
            # plan due to the addition of ControlNet, e.g., Z-Image-ControlNet.
            parallelism_config.parallel_kwargs["has_controlnet"] = _has_controlnet(
                pipe_or_adapter,
            )
            parallelism_config._has_controlnet = parallelism_config.parallel_kwargs[
                "has_controlnet"
            ]

        # Parse extra parallel modules from names to actual modules
        if (
            extra_parallel_module := parallelism_config.parallel_kwargs.get(
                "extra_parallel_modules", None
            )
        ) is not None:
            parallelism_config.parallel_kwargs["extra_parallel_modules"] = (
                _parse_extra_parallel_modules(
                    pipe_or_adapter,
                    extra_parallel_module,
                )
            )

        transformers = []
        if isinstance(pipe_or_adapter, DiffusionPipeline):
            adapter = BlockAdapterRegister.get_adapter(
                pipe_or_adapter, skip_post_init=cache_config is None
            )
            if adapter is None:
                assert hasattr(pipe_or_adapter, "transformer"), (
                    "The given DiffusionPipeline does not have "
                    "a 'transformer' attribute, cannot enable "
                    "parallelism."
                )
                transformers = [pipe_or_adapter.transformer]
            else:
                adapter = BlockAdapter.normalize(adapter, unique=False)
                transformers = BlockAdapter.flatten(adapter.transformer)
        else:
            if not BlockAdapter.is_normalized(pipe_or_adapter):
                pipe_or_adapter = BlockAdapter.normalize(pipe_or_adapter, unique=False)
            transformers = BlockAdapter.flatten(pipe_or_adapter.transformer)

        if len(transformers) == 0:
            logger.warning(
                "No transformer is detected in the " "BlockAdapter, skip enabling parallelism."
            )
            return pipe_or_adapter

        if len(transformers) > 1:
            logger.warning(
                "Multiple transformers are detected in the "
                "BlockAdapter, all transfomers will be "
                "enabled for parallelism."
            )
        for i, transformer in enumerate(transformers):
            # Enable parallelism for the transformer inplace
            transformers[i] = enable_parallelism(transformer, parallelism_config)
    return pipe_or_adapter


def _has_controlnet(pipe_or_adapter: DiffusionPipeline | BlockAdapter) -> bool:
    """Check if the given pipeline has ControlNet."""
    if isinstance(pipe_or_adapter, BlockAdapter):
        pipe = pipe_or_adapter.pipe
    else:
        pipe = pipe_or_adapter
    if hasattr(pipe, "controlnet") and getattr(pipe, "controlnet") is not None:
        return True
    return False


def _parse_text_encoder(
    pipe: DiffusionPipeline,
) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    pipe_cls_name = pipe.__class__.__name__
    if (
        hasattr(pipe, "text_encoder_2")
        and not pipe_cls_name.startswith("Hunyuan")
        and not pipe_cls_name.startswith("Kandinsky")
    ):
        # Specific for FluxPipeline, FLUX.1-dev
        return getattr(pipe, "text_encoder_2"), "text_encoder_2"
    elif hasattr(pipe, "text_encoder_3"):  # HiDream pipeline
        return getattr(pipe, "text_encoder_3"), "text_encoder_3"
    elif hasattr(pipe, "text_encoder"):  # General case
        return getattr(pipe, "text_encoder"), "text_encoder"
    else:
        return None, None


def _parse_extra_parallel_modules(
    pipe_or_adapter: DiffusionPipeline | BlockAdapter,
    extra_parallel_module: List[str | torch.nn.Module],
) -> Union[List[torch.nn.Module], List]:
    if isinstance(pipe_or_adapter, BlockAdapter):
        pipe = pipe_or_adapter.pipe
    else:
        pipe = pipe_or_adapter

    if not extra_parallel_module:  # empty list
        return []

    parsed_extra_parallel_modules: List[torch.nn.Module] = []
    for module_or_name in extra_parallel_module:
        if isinstance(module_or_name, torch.nn.Module):
            parsed_extra_parallel_modules.append(module_or_name)
            continue

        if hasattr(pipe, module_or_name):
            if module_or_name == "text_encoder":
                # Special handling for text encoder
                text_encoder, _ = _parse_text_encoder(pipe)
                if text_encoder is not None:
                    parsed_extra_parallel_modules.append(text_encoder)
                else:
                    logger.warning(
                        "Text encoder not found in the pipeline for extra parallel module."
                    )
            else:
                parsed_extra_parallel_modules.append(getattr(pipe, module_or_name))
        else:
            logger.warning(
                f"Extra parallel module name {module_or_name} not found in the pipeline."
            )
    return parsed_extra_parallel_modules


def set_attn_backend(
    pipe_or_adapter: Union[DiffusionPipeline, BlockAdapter],
    attention_backend: Optional[str] = None,
):
    if attention_backend is None:
        return

    # non-parallelism or non-cache case: set attention backend directly
    try:
        from ..parallelism.attention import _maybe_register_custom_attn_backends

        _maybe_register_custom_attn_backends()
    except Exception as e:
        logger.warning(
            "Failed to register custom attention backends. "
            f"Proceeding to set attention backend anyway. Error: {e}"
        )

    def _set_backend(module):
        if module is None:
            return
        if hasattr(module, "set_attention_backend") and isinstance(module, ModelMixin):
            module.set_attention_backend(attention_backend)
            logger.info(
                f"Set attention backend to {attention_backend} for module: {module.__class__.__name__}."
            )
        else:
            logger.warning(
                "--attn was provided but module does not support set_attention_backend: "
                f"{module.__class__.__name__}."
            )

    try:
        if isinstance(pipe_or_adapter, BlockAdapter):
            transformer = pipe_or_adapter.transformer
            if isinstance(transformer, list):
                for t in transformer:
                    _set_backend(t)
            else:
                _set_backend(transformer)
        else:
            pipe = pipe_or_adapter
            if hasattr(pipe, "transformer"):
                _set_backend(getattr(pipe, "transformer"))
            else:
                _set_backend(pipe)
    except Exception as e:
        raise RuntimeError(
            f"Failed to set attention backend to {attention_backend}. "
            "This usually means the backend is unavailable (e.g., FlashAttention-3 not installed) "
            "or the model/shape/dtype is unsupported. "
            f"Original error: {e}"
        ) from e


def refresh_context(
    transformer: torch.nn.Module,
    **force_refresh_kwargs,
):
    r"""Refresh cache context for the given transformer. This is useful when
    the users run into transformer-only case with dynamic num_inference_steps.
    For example, when num_inference_steps changes significantly between different
    requests, the cache context should be refreshed to avoid potential
    precision degradation. Usage:
    ```py
    >>> import cache_dit
    >>> from cache_dit import DBCacheConfig
    >>> from diffusers import DiffusionPipeline
    >>> # Init cache context with num_inference_steps=None (default)
    >>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
    >>> pipe = cache_dit.enable_cache(pipe.transformer, cache_config=DBCacheConfig(...))
    >>> # Assume num_inference_steps is 28, and we want to refresh the context
    >>> cache_dit.refresh_context(transformer, num_inference_steps=28, verbose=True)
    >>> output = pipe(...) # Just call the pipe as normal.
    >>> stats = cache_dit.summary(pipe.transformer) # Then, get the summary
    >>> # Update the cache context with new num_inference_steps=50.
    >>> cache_dit.refresh_context(pipe.transformer, num_inference_steps=50, verbose=True)
    >>> output = pipe(...) # Just call the pipe as normal.
    >>> stats = cache_dit.summary(pipe.transformer) # Then, get the summary
    >>> # Update the cache context with new cache_config.
    >>> cache_dit.refresh_context(
        pipe.transformer,
        cache_config=DBCacheConfig(
            residual_diff_threshold=0.1,
            max_warmup_steps=10,
            max_cached_steps=20,
            max_continuous_cached_steps=4,
            num_inference_steps=50,
        ),
        verbose=True,
    )
    >>> output = pipe(...) # Just call the pipe as normal.
    >>> stats = cache_dit.summary(pipe.transformer) # Then, get the summary
    ```
    """
    if force_refresh_kwargs:
        if "cache_config" not in force_refresh_kwargs:
            # Assume force_refresh_kwargs is passed as dict, e.g.,
            # {"num_inference_steps": 50}
            from .utils import load_cache_config

            cache_config, calibrator_config = load_cache_config(
                force_refresh_kwargs,
                reset=True,
            )
            force_refresh_kwargs["cache_config"] = copy.deepcopy(cache_config)
            if calibrator_config is not None:
                force_refresh_kwargs["calibrator_config"] = copy.deepcopy(calibrator_config)
        else:
            allowed_keys = {"cache_config", "calibrator_config", "verbose"}
            not_allowed_keys = set(force_refresh_kwargs.keys()) - allowed_keys
            if not_allowed_keys:
                logger.warning(
                    f"force_refresh_kwargs contains cache_config, please put the extra "
                    f"kwargs: {not_allowed_keys} into cache_config directly. Ohtherwise, "
                    f"these kwargs will be ignored."
                )
    CachedAdapter.maybe_refresh_context(
        transformer,
        **force_refresh_kwargs,
    )


def disable_cache(
    pipe_or_adapter: Union[
        DiffusionPipeline,
        BlockAdapter,
        torch.nn.Module,  # Transformer-only
    ],
):
    cls_name = pipe_or_adapter.__class__.__name__
    CachedAdapter.maybe_release_hooks(pipe_or_adapter)
    logger.warning(f"Acceleration hooks is disabled for: {cls_name}.")


def supported_pipelines(
    **kwargs,
) -> Tuple[int, List[str]]:
    return BlockAdapterRegister.supported_pipelines(**kwargs)


def get_adapter(
    pipe: DiffusionPipeline | str | Any,
) -> BlockAdapter:
    return BlockAdapterRegister.get_adapter(pipe)


def _steps_mask(
    compute_bins: List[int],
    cache_bins: List[int],
    total_steps: Optional[int] = None,
) -> list[int]:
    mask = []
    step = 0
    compute_bins = compute_bins.copy()
    cache_bins = cache_bins.copy()
    # reverse to use as stacks
    compute_bins.reverse()
    cache_bins.reverse()

    if total_steps is not None:
        assert (
            sum(compute_bins) + sum(cache_bins) >= total_steps
        ), "The sum of compute and cache intervals must be at least total_steps."
    else:
        total_steps = sum(compute_bins) + sum(cache_bins)

    while step < total_steps:

        if compute_bins:
            ci = compute_bins.pop()
            mask.extend([1] * ci)
            step += ci
        if cache_bins:
            cai = cache_bins.pop()
            mask.extend([0] * cai)
            step += cai

        if step >= total_steps:
            break

    return mask[:total_steps]


def steps_mask(
    compute_bins: Optional[List[int]] = None,
    cache_bins: Optional[List[int]] = None,
    total_steps: Optional[int] = None,
    mask_policy: Optional[str] = "medium",
) -> list[int]:
    r"""
    Define a step computation mask based on compute and cache bins.

    Args:
        compute_bins (`List[int]`, *optional*, defaults to None):
            A list specifying the number of consecutive steps to compute.
            For example, [4, 2] means compute 4 steps, then 2 steps.
        cache_bins (`List[int]`, *optional*, defaults to None):
            A list specifying the number of consecutive steps to cache.
            For example, [2, 4] means cache 2 steps, then 4 steps.
        total_steps (`int`, *optional*, defaults to None):
            Total number of steps for which the mask is generated.
            If provided, the sum of compute_bins and cache_bins must be at
            least total_steps.
        mask_policy (`str`, *optional*, defaults to "medium"):
            Predefined mask policy. Options are "slow", "medium", "fast", "ultra".
            For examples, if total_steps=28, each policy corresponds to specific
            compute and cache bin configurations:
                - "slow": compute_bins=[8, 3, 3, 2, 1, 1], cache_bins=1, 2, 2, 2, 3]
                - "medium": compute_bins=[6, 2, 2, 2, 2, 1], cache_bins=[1, 3, 3, 3, 3]
                - "fast": compute_bins=[6, 1, 1, 1, 1], cache_bins=[1, 3, 4, 5, 4]
                - "ultra": compute_bins=[4, 1, 1, 1, 1], cache_bins=[2, 5, 6, 7]
    Returns:
        `List[int]`: A list representing the step computation mask, where 1
        indicates a compute step and 0 indicates a cache step.
    """
    # Prefer compute/cache bins if both are provided
    if compute_bins is not None and cache_bins is not None:
        return _steps_mask(
            compute_bins=compute_bins,
            cache_bins=cache_bins,
            total_steps=total_steps,
        )

    assert (
        total_steps is not None
    ), "total_steps must be provided when using predefined mask_policy."
    # 28 steps predefined policies
    predefined_policies = {
        # NOTE: last step will never cache by default
        # mask: 11111111 0 111 00 111 00 11 00 1 000 1
        "slow": [
            [8, 3, 3, 2, 1, 1],  # = 18 compute steps
            [1, 2, 2, 2, 3],  # = 10 cache steps
        ],
        "medium": [
            [6, 2, 2, 2, 2, 1],  # = 15 compute steps
            [1, 3, 3, 3, 3],  # = 13 cache steps
        ],
        "fast": [
            [6, 1, 1, 1, 1, 1],  # = 11 compute steps
            [1, 3, 4, 5, 4],  # = 17 cache steps
        ],
        "ultra": [
            [4, 1, 1, 1, 1],  # = 8 compute steps
            [2, 5, 6, 7],  # = 20 cache steps
        ],
    }

    def _sum_policy(policy: List[List[int]]) -> int:
        return sum(policy[0]) + sum(policy[1])

    def _truncate_policy(policy: List[List[int]], target_steps: int) -> List[List[int]]:
        compute_bins, cache_bins = policy  # reference only
        while _sum_policy(policy) > target_steps:
            if cache_bins:
                cache_bins[-1] -= 1
                if cache_bins[-1] == 0:
                    cache_bins.pop()
            if _sum_policy(policy) <= target_steps:
                break
            if compute_bins:
                compute_bins[-1] -= 1
                if compute_bins[-1] == 0:
                    compute_bins.pop()
            if _sum_policy(policy) <= target_steps:
                break
        return [compute_bins, cache_bins]

    def _truncate_predefined_policies(
        policies: dict[str, List[List[int]]],
        target_steps: int,
    ) -> dict[str, List[List[int]]]:
        truncated_policies = {}
        for name, policy in policies.items():
            truncated_policies[name] = _truncate_policy(policy, target_steps)
        return truncated_policies

    if total_steps > 28:
        # Expand bins if total_steps exceed predefined sum
        # For example, for total_steps=50, we will expand the bins
        # of each policy until they can cover total_steps.
        # This ensures the relative ratio of compute/cache steps
        # remains consistent with the predefined policies.
        for policy in predefined_policies.values():
            min_bins_len = min(len(policy[0]), len(policy[1]))
            compute_bins = copy.deepcopy(policy[0])
            cache_bins = copy.deepcopy(policy[1])
            while _sum_policy(policy) < total_steps:
                for i in range(min_bins_len):
                    # Add 1 to each compute bin, e.g., total_steps=50,
                    # slow: 8 -> 8 + int(8 * (50 / 28) * 0.5) = 14
                    #       3 -> 3 + int(3 * (50 / 28) * 0.5) = 5
                    # fast: 6 -> 6 + int(6 * (50 / 28) * 0.5) = 11
                    #       1 -> 1 + int(1 * (50 / 28) * 0.5) = 2
                    policy[0][i] += max(int(compute_bins[i] * ((total_steps / 28) * 0.5)), 1)
                    if _sum_policy(policy) >= total_steps:
                        break
                    # Add 1 to each cache bin, e.g., total_steps=50,
                    # slow: 1 -> 1 + int(1 * (50 / 28) * 0.5) = 2
                    #       2 -> 2 + int(2 * (50 / 28) * 0.5) = 4
                    # fast: 1 -> 1 + int(1 * (50 / 28) * 0.5) = 2
                    #       3 -> 3 + int(3 * (50 / 28) * 0.5) = 5
                    policy[1][i] += max(int(cache_bins[i] * ((total_steps / 28) * 0.5)), 1)
                    if _sum_policy(policy) >= total_steps:
                        break
                if _sum_policy(policy) >= total_steps:
                    break
                # compute bin due to compute_bins always longer than cache_bins
                policy[0][-1] += 1
                if _sum_policy(policy) >= total_steps:
                    break

        # truncate to exact total_steps
        predefined_policies = _truncate_predefined_policies(
            predefined_policies,
            total_steps,
        )

    elif total_steps < 28 and total_steps >= 16:
        # Truncate bins to fit total_steps
        predefined_policies = _truncate_predefined_policies(
            predefined_policies,
            total_steps,
        )
    elif total_steps < 16 and total_steps >= 8:
        # Mainly for distilled models with less steps, use smaller compute/cache bins
        if total_steps > 8:
            predefined_policies = {
                "slow": [
                    [4, 2, 2, 2, 1],  # = 11
                    [1, 1, 1, 1],  # = 4
                ],
                "medium": [
                    [4, 2, 1, 1, 1],  # = 9
                    [1, 1, 2, 2],  # = 6
                ],
                "fast": [
                    [3, 1, 1, 1, 1],  # = 7
                    [1, 2, 2, 3],  # = 8
                ],
                "ultra": [
                    [2, 1, 1, 1, 1],  # = 6
                    [1, 2, 3, 3],  # = 9
                ],
            }
            # Specifical case for Z-Image-Turbo with 9 steps
            if total_steps == 9:
                predefined_policies = {
                    "slow": [
                        [5, 2, 1],  # = 8
                        [1],  # = 1
                    ],
                    "medium": [
                        [5, 1, 1],  # = 7
                        [1, 1],  # = 2
                    ],
                    "fast": [
                        [4, 1, 1],  # = 6
                        [1, 2],  # = 3
                    ],
                    "ultra": [
                        [3, 1, 1],  # = 5
                        [2, 2],  # = 4
                    ],
                }
        else:  # total_steps == 8
            # cases: 8 steps distilled models
            predefined_policies = {
                "slow": [
                    [5, 1, 1],  # = 7
                    [1],  # = 1
                ],
                "medium": [
                    [4, 1, 1],  # = 6
                    [1, 1],  # = 2
                ],
                "fast": [
                    [3, 1, 1],  # = 5
                    [1, 2],  # = 3
                ],
                "ultra": [
                    [2, 1, 1],  # = 4
                    [2, 2],  # = 4
                ],
            }
        for policy in predefined_policies.values():
            predefined_policies = _truncate_predefined_policies(
                predefined_policies,
                total_steps,
            )
    elif total_steps < 8:
        # case: 4 or 6 steps distilled models
        assert total_steps in (4, 6), (
            "Only total_steps=4 or 6 is supported for predefined masks "
            f"while total_steps < 8. Got total_steps={total_steps}."
        )
        constant_plicy_4_steps = [[2, 1], [1]]
        constant_plicy_6_steps = [[3, 1], [2]]
        if total_steps == 4:
            constant_plicy = constant_plicy_4_steps
        else:
            constant_plicy = constant_plicy_6_steps

        predefined_policies = {
            "slow": constant_plicy,
            "medium": constant_plicy,
            "fast": constant_plicy,
            "ultra": constant_plicy,
        }

    if mask_policy not in predefined_policies:
        raise ValueError(
            f"mask_policy {mask_policy} is not valid. "
            f"Choose from {list(predefined_policies.keys())}."
        )
    compute_bins, cache_bins = predefined_policies[mask_policy]
    # Will truncate if exceeded total_steps
    compute_mask = _steps_mask(
        compute_bins=compute_bins, cache_bins=cache_bins, total_steps=total_steps
    )
    # Force last step to compute
    compute_mask[-1] = 1
    return compute_mask
