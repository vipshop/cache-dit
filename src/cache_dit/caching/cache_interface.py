import copy
import torch
from typing import Any, Tuple, List, Union, Optional
from diffusers import DiffusionPipeline, ModelMixin
from .cache_types import CacheType
from .block_adapters import BlockAdapter
from .block_adapters import BlockAdapterRegister
from .cache_adapters import CachedAdapter
from .cache_contexts import DBCacheConfig
from .cache_contexts import DBPruneConfig
from .cache_contexts import CalibratorConfig
from .params_modifier import ParamsModifier
from ..parallelism import ParallelismConfig
from ..parallelism import enable_parallelism
from ..quantization import QuantizeConfig
from ..quantization import quantize
from ..utils import check_controlnet
from ..utils import parse_extra_modules
from ..logger import init_logger

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
  cache_config: Optional[Union[
    DBCacheConfig,
    DBPruneConfig,
  ]] = None,
  # Calibrator config: TaylorSeerCalibratorConfig, etc.
  calibrator_config: Optional[CalibratorConfig] = None,
  # Modify cache context params for specific blocks.
  params_modifiers: Optional[Union[
    ParamsModifier,
    List[ParamsModifier],
    List[List[ParamsModifier]],
  ]] = None,
  # Config for Parallelism
  parallelism_config: Optional[ParallelismConfig] = None,
  # Allow set custom attention backend for non-parallelism case
  attention_backend: Optional[str] = None,
  # Quantize config
  quantize_config: Optional[QuantizeConfig] = None,
  # Other cache context kwargs: Deprecated cache kwargs
  **kwargs,
) -> Union[
    DiffusionPipeline,
    # Transformer-only
    torch.nn.Module,
    ModelMixin,
    BlockAdapter,
]:
  """Enable cache acceleration for a pipeline, adapter, or transformer module.

  This is the main cache-dit entry point for wiring DBCache into a model. It can wrap a full
  `DiffusionPipeline`, a normalized `BlockAdapter`, or a transformer module directly, and it can
  optionally layer parallelism, attention-backend selection, and quantization on top of the cache
  hooks.

  Use `cache_config` to control the caching strategy itself. Detailed cache-policy fields such as
  warmup behavior, residual thresholds, or step masks are documented on `DBCacheConfig` and
  `DBPruneConfig`. Calibrator-specific options belong on `calibrator_config`, and distributed
  execution options belong on `parallelism_config`.

  :param pipe_or_adapter: Pipeline, adapter, or transformer module to accelerate.
  :param cache_config: Cache configuration describing the runtime caching strategy. When omitted and
    no other optimization config is provided, a default `DBCacheConfig` is created.
  :param calibrator_config: Optional calibrator configuration used to augment the cache context.
  :param params_modifiers: Optional per-block cache-context overrides applied after the base config
    is loaded.
  :param parallelism_config: Optional parallelism configuration applied after cache hooks are
    installed.
  :param attention_backend: Optional attention backend override for non-parallel execution.
  :param quantize_config: Optional quantization configuration applied after cache and parallelism.
  :param kwargs: Deprecated cache-context keyword arguments kept for backward compatibility.
  :returns: The original object with cache-dit hooks and optional optimizations applied.

  Example::

    >>> import cache_dit
    >>> from diffusers import DiffusionPipeline
    >>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
    >>> pipe = cache_dit.enable_cache(pipe)
    >>> output = pipe(...)
    >>> stats = cache_dit.summary(pipe)
  """
  # Precheck for compatibility of different configurations
  if cache_config is None:
    # Allow empty cache_config when other optimization configs are provided, but
    # log a info to remind users to set up cache_config later for better performance.
    # We will set default cache config only when all configs are None (e.g, all configs:
    # parallelism_config, attention_backend, cache_config, quantize_config are None).
    if parallelism_config is None and attention_backend is None and quantize_config is None:
      # Set default cache config only when parallelism is not enabled
      logger.info("cache_config is None, using default DBCacheConfig")
      cache_config = DBCacheConfig()

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
    logger.warning("Manually settup DBCache context without DBCacheConfig is "
                   "deprecated and will be removed in the future, please use "
                   "`cache_config` parameter instead!")
    if cache_config is not None:
      cache_config.update(**deprecated_kwargs)
    else:
      cache_config = DBCacheConfig(**deprecated_kwargs)

  if cache_config is not None:
    context_kwargs["cache_config"] = cache_config

  # NOTE: Deprecated taylorseer params. These parameters are now retained
  # for backward compatibility but will be removed in the future.
  if cache_config is not None and (kwargs.get("enable_taylorseer", None) is not None
                                   or kwargs.get("enable_encoder_taylorseer", None) is not None):
    logger.warning("Manually settup TaylorSeer calibrator without TaylorSeerCalibratorConfig is "
                   "deprecated and will be removed in the future, please use "
                   "`calibrator_config` parameter instead!")
    from .cache_contexts.calibrators import TaylorSeerCalibratorConfig

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
      raise ValueError(f"type: {type(pipe_or_adapter)} is not valid, "
                       "Please pass DiffusionPipeline or BlockAdapter"
                       "for the 1's position param: pipe_or_adapter")
  else:
    logger.warning("cache_config is None, skip cache acceleration for "
                   f"{pipe_or_adapter.__class__.__name__}.")

  # Set custom attention backend for non-parallelism case
  if attention_backend is not None:
    if parallelism_config is not None:
      if parallelism_config.attention_backend is not None:
        logger.warning("Both attention_backend in parallelism_config and "
                       "attention_backend param are provided, prefer using "
                       "attention_backend in parallelism_config.")
        attention_backend = None  # Prefer parallelism_config setting
      else:
        logger.info("Setting attention_backend from attention_backend "
                    "param to parallelism_config.")
        parallelism_config.attention_backend = attention_backend
    else:
      set_attn_backend(pipe_or_adapter, attention_backend)

  # NOTE: Users should always enable parallelism/quantization after applying
  # cache to avoid hooks conflict.
  transformers = []
  if parallelism_config is not None or quantize_config is not None:
    if isinstance(pipe_or_adapter, DiffusionPipeline):
      adapter = BlockAdapterRegister.get_adapter(pipe_or_adapter,
                                                 skip_post_init=cache_config is None)
      if adapter is None:
        assert hasattr(pipe_or_adapter,
                       "transformer"), ("The given DiffusionPipeline does not have "
                                        "a 'transformer' attribute, cannot enable "
                                        "parallelism.")
        transformers = [pipe_or_adapter.transformer]
      else:
        adapter = BlockAdapter.normalize(adapter, unique=False)
        transformers = BlockAdapter.flatten(adapter.transformer)
    else:
      if not BlockAdapter.is_normalized(pipe_or_adapter):
        pipe_or_adapter = BlockAdapter.normalize(pipe_or_adapter, unique=False)
      transformers = BlockAdapter.flatten(pipe_or_adapter.transformer)

    if len(transformers) == 0:
      logger.warning("No transformer is detected in the BlockAdapter, skip enabling "
                     "parallelism or quantization.")
      return pipe_or_adapter

    if len(transformers) > 1:
      logger.warning("Multiple transformers are detected in the BlockAdapter, all "
                     "transformers will be enabled for parallelism or quantization.")

  pipe = (pipe_or_adapter if isinstance(pipe_or_adapter, DiffusionPipeline) else getattr(
    pipe_or_adapter, "pipe", None))

  # Enable parallelism if parallelism_config is provided.
  if parallelism_config is not None:
    assert isinstance(parallelism_config,
                      ParallelismConfig), "parallelism_config should be of type ParallelismConfig."

    # Prefer custom has_controlnet flag from users if provided, otherwise,
    # we will automatically check whether the pipeline has controlnet.
    if not parallelism_config._has_controlnet:
      # This flag is used to decide whether to use the special parallelism
      # plan due to the addition of ControlNet, e.g., Z-Image-ControlNet.
      parallelism_config._has_controlnet = check_controlnet(pipe)

    # Parse extra parallel modules from names to actual modules
    extra_parallel_module = parallelism_config.extra_parallel_modules
    if extra_parallel_module is not None and pipe is not None:
      parallelism_config.extra_parallel_modules = parse_extra_modules(pipe, extra_parallel_module)

    for i, transformer in enumerate(transformers):
      # Enable parallelism for the transformer inplace
      transformers[i] = enable_parallelism(transformer, parallelism_config)

  # Enable quantization if quantize_config is provided.
  if quantize_config is not None:
    assert isinstance(quantize_config,
                      QuantizeConfig), "quantize_config should be of type QuantizeConfig."

    # By default, we will try to apply quantization to transformer module(s)
    # for better performance. User can specify the quantization modules more
    # precisely with quantize_config.components_to_quantize. For example,
    # when quantize_config.components_to_quantize is set to ['transformer',
    # 'text_encoder'], we will apply quantization to both transformer and
    # text encoder modules with the specified quantization type.
    if quantize_config.components_to_quantize is None:
      for i, transformer in enumerate(transformers):
        # Enable quantization for the transformer inplace
        transformers[i] = quantize(transformer, quantize_config=quantize_config)
    else:
      # Expand the quantize_config with multiple components to multiple simple
      # configs with single component.
      if pipe is not None:
        expanded_quantize_configs = QuantizeConfig.expand_configs(quantize_config)
        for config in expanded_quantize_configs:
          components_to_quantize = config.components_to_quantize
          components = parse_extra_modules(pipe, components_to_quantize)
          assert len(components) == len(components_to_quantize), (
            f"Some components in quantize_config.components_to_quantize: "
            f"{components_to_quantize} are not found in the pipeline, please check the "
            f"component names or directly pass the actual modules in components_to_quantize.")
          for component, name in zip(components, components_to_quantize):
            # The text_encoder module maybe overridden by parse_extra_modules,
            # so we will try to get the actual module name from the patched
            # _actual_module_name attribute if exists.
            name = getattr(component, "_actual_module_name", name)
            # Enable quantization for the specified component inplace
            quantized_component = quantize(component, quantize_config=config)
            setattr(pipe, name, quantized_component)
  return pipe_or_adapter


def set_attn_backend(
  pipe_or_adapter: Union[DiffusionPipeline, BlockAdapter],
  attention_backend: Optional[str] = None,
):
  """Set the attention backend on a pipeline or normalized adapter.

  :param pipe_or_adapter: Pipeline or adapter whose transformer modules should be updated.
  :param attention_backend: Attention backend name to apply. When omitted, the function returns
    without making changes.
  """

  if attention_backend is None:
    return

  # non-parallelism or non-cache case: set attention backend directly
  try:
    from ..parallelism.attention import _maybe_register_custom_attn_backends

    _maybe_register_custom_attn_backends()
  except Exception as e:
    logger.warning("Failed to register custom attention backends. "
                   f"Proceeding to set attention backend anyway. Error: {e}")

  def _set_backend(module):
    if module is None:
      return
    if hasattr(module, "set_attention_backend") and isinstance(module, ModelMixin):
      module.set_attention_backend(attention_backend)
      logger.info(
        f"Set attention backend to <{attention_backend}> for module: {module.__class__.__name__}.")
    else:
      logger.warning("--attn was provided but module does not support set_attention_backend: "
                     f"{module.__class__.__name__}.")

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
      f"Failed to set attention backend to <{attention_backend}>. "
      "This usually means the backend is unavailable (e.g., FlashAttention-3 not installed) "
      "or the model/shape/dtype is unsupported. "
      f"Original error: {e}") from e


def refresh_context(
  transformer: torch.nn.Module,
  **force_refresh_kwargs,
):
  """Refresh the cache context attached to a transformer module.

  This helper is mainly useful for transformer-only flows where runtime conditions such as
  `num_inference_steps` change across requests and the cached context should be rebuilt without
  re-running `enable_cache` from scratch.

  Example::

    >>> import cache_dit
    >>> from cache_dit import DBCacheConfig
    >>> from diffusers import DiffusionPipeline
    >>> # Init cache context with num_inference_steps=None (default)
    >>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
    >>> pipe = cache_dit.enable_cache(pipe.transformer, cache_config=DBCacheConfig(...))
    >>> # Assume num_inference_steps is 28, and we want to refresh the context
    >>> cache_dit.refresh_context(pipe.transformer, num_inference_steps=28, verbose=True)
    >>> output = pipe(...)
    >>> stats = cache_dit.summary(pipe.transformer)
    >>> # Update the cache context with new num_inference_steps=50.
    >>> cache_dit.refresh_context(pipe.transformer, num_inference_steps=50, verbose=True)
    >>> output = pipe(...)
    >>> stats = cache_dit.summary(pipe.transformer)
    >>> # Update the cache context with new cache_config.
    >>> cache_dit.refresh_context(
    ...   pipe.transformer,
    ...   cache_config=DBCacheConfig(
    ...     residual_diff_threshold=0.1,
    ...     max_warmup_steps=10,
    ...     max_cached_steps=20,
    ...     max_continuous_cached_steps=4,
    ...     num_inference_steps=50,
    ...   ),
    ...   verbose=True,
    ... )
    >>> output = pipe(...)
    >>> stats = cache_dit.summary(pipe.transformer)

  :param transformer: Transformer module previously passed to `enable_cache`.
  :param force_refresh_kwargs: Either a full `cache_config` / `calibrator_config` pair or shorthand
    cache fields such as `num_inference_steps` that should be reloaded into a fresh context.
  """
  if force_refresh_kwargs:
    if "cache_config" not in force_refresh_kwargs:
      # Assume force_refresh_kwargs is passed as dict, e.g.,
      # {"num_inference_steps": 50}
      from .load_configs import load_cache_config

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
        logger.warning(f"force_refresh_kwargs contains cache_config, please put the extra "
                       f"kwargs: {not_allowed_keys} into cache_config directly. Otherwise, "
                       f"these kwargs will be ignored.")
  CachedAdapter.maybe_refresh_context(
    transformer,
    **force_refresh_kwargs,
  )


def disable_cache(
  pipe_or_adapter: Union[
    DiffusionPipeline,
    BlockAdapter,
    torch.nn.Module,  # Transformer-only
  ], ):
  """Release cache hooks and restore the original uncached forward path.

  :param pipe_or_adapter: Pipeline or adapter to process.
  """

  cls_name = pipe_or_adapter.__class__.__name__
  CachedAdapter.maybe_release_hooks(pipe_or_adapter)
  logger.warning(f"Acceleration hooks is disabled for: {cls_name}.")


def supported_pipelines(**kwargs, ) -> Tuple[int, List[str]]:
  """Return the number and names of pipelines with registered adapters.

  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A tuple of `(count, pipeline_names)` describing the registered adapters.
  """

  return BlockAdapterRegister.supported_pipelines(**kwargs)


def get_adapter(pipe: DiffusionPipeline | str | Any, ) -> BlockAdapter:
  """Resolve a registered `BlockAdapter` for a pipeline instance or name.

  :param pipe: Pipeline instance or registered pipeline name.
  :returns: The resolved adapter.
  """

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
    assert (sum(compute_bins) + sum(cache_bins)
            >= total_steps), "The sum of compute and cache intervals must be at least total_steps."
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
  """Define a step computation mask based on compute and cache bins.

  :param compute_bins: Consecutive compute intervals. For example, `[4, 2]` means compute four
    steps, then later compute two more steps.
  :param cache_bins: Consecutive cache intervals paired with `compute_bins`.
  :param total_steps: Total number of steps to emit. When provided, the sum of the bin lengths must
    cover at least this many steps.
  :param mask_policy: Predefined mask policy used when explicit bins are not provided. Supported
    values are `"slow"`, `"medium"`, `"fast"`, and `"ultra"`.
  :returns: A list where `1` means compute the step and `0` means reuse cache.
  """
  # Prefer compute/cache bins if both are provided
  if compute_bins is not None and cache_bins is not None:
    return _steps_mask(
      compute_bins=compute_bins,
      cache_bins=cache_bins,
      total_steps=total_steps,
    )

  assert (total_steps
          is not None), "total_steps must be provided when using predefined mask_policy."
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
    assert total_steps in (4, 6), ("Only total_steps=4 or 6 is supported for predefined masks "
                                   f"while total_steps < 8. Got total_steps={total_steps}.")
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
    raise ValueError(f"mask_policy {mask_policy} is not valid. "
                     f"Choose from {list(predefined_policies.keys())}.")
  compute_bins, cache_bins = predefined_policies[mask_policy]
  # Will truncate if exceeded total_steps
  compute_mask = _steps_mask(compute_bins=compute_bins,
                             cache_bins=cache_bins,
                             total_steps=total_steps)
  # Force last step to compute
  compute_mask[-1] = 1
  return compute_mask
