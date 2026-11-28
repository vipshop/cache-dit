"""Top-level optimization pipeline: cache + parallelism, quantization, compile and offload."""

import torch.distributed as dist

from ..attention import set_attn_backend
from ..caching import (
  BlockAdapter,
  DBCacheConfig,
  enable_cache,
  load_configs,
  steps_mask,
)
from ..distributed import ParallelismBackend, ParallelismConfig
from ..logger import init_logger
from .cache import build_calibrator_config
from .compile import (
  maybe_compile_controlnet,
  maybe_compile_text_encoder,
  maybe_compile_transformer,
  maybe_compile_vae,
)
from .offload import (
  maybe_adaln_host_residency,
  maybe_cpu_offload,
  maybe_generic_module_offload,
  maybe_vae_tiling_or_slicing,
)
from .parallel import get_rank_device, prepare_extra_parallel_modules
from .quantize import (
  _register_deferred_few_shot_pipeline_move_callbacks,
  _should_skip_initial_few_shot_pipeline_move,
  maybe_quantize_text_encoder,
  maybe_quantize_transformer,
)

logger = init_logger(__name__)


def maybe_apply_optimization(
  args,
  pipe_or_adapter,
  **kwargs,
):
  quantize_config = None
  default_num_inference_steps = kwargs.pop("default_num_inference_steps", None)
  if args.cache or args.parallel_type is not None or args.config_path is not None:

    if args.config_path is None:
      # Construct acceleration configs from command line args if config path is not provided
      cache_config = kwargs.pop("cache_config", None)
      parallelism_config = kwargs.pop("parallelism_config", None)

      extra_parallel_modules = prepare_extra_parallel_modules(
        args,
        pipe_or_adapter,
        custom_extra_modules=kwargs.get("extra_parallel_modules", None),
      )

      # Caching and Parallelism
      if args.steps_mask and args.mask_policy is not None:
        logger.info(f"Using steps computation mask with policy: {args.mask_policy} for caching.")
        if default_num_inference_steps is None:
          assert (args.num_inference_steps
                  is not None), "num_inference_steps (--steps) must be provided for steps mask."
          num_inference_steps = args.num_inference_steps
        else:
          num_inference_steps = default_num_inference_steps
        steps_computation_mask = steps_mask(
          total_steps=num_inference_steps,
          mask_policy=args.mask_policy,
        )
      else:
        steps_computation_mask = None

      def _prepare_distributed_size():
        if args.parallel_type is not None:
          world_size = dist.get_world_size() if dist.is_initialized() else 1
          degrees = args.parallel_degrees
          product = degrees["ulysses"] * degrees["ring"] * degrees["tp"]
          assert product == world_size, (
            f"--parallel-params degrees (ulysses={degrees['ulysses']},ring={degrees['ring']},"
            f"tp={degrees['tp']}) multiply to {product}, which must equal the world size "
            f"{world_size}.")
          # Inactive axes stay None so the pipeline does not build 1-rank groups for them.
          return tuple(degrees[axis] if axis in args.parallel_axes else None
                       for axis in ("ulysses", "ring", "tp"))
        return None, None, None

      ulysses_size, ring_size, tp_size = _prepare_distributed_size()

      if cache_config is not None and args.seacache:
        cache_config.update(
          enable_seacache=True,
          seacache_mode=args.seacache_mode,
          seacache_feature_shape=args.seacache_feature_shape,
          seacache_power_exp=args.seacache_power_exp,
        )
        if args.softcache:
          cache_config.update(
            enable_softcache=True,
            softcache_tau=args.softcache_tau,
            softcache_jf_max=args.softcache_jf_max,
          )

      enable_cache(
        pipe_or_adapter,
        cache_config=(DBCacheConfig(
          Fn_compute_blocks=args.Fn_compute_blocks,
          Bn_compute_blocks=args.Bn_compute_blocks,
          max_warmup_steps=args.max_warmup_steps,
          warmup_interval=args.warmup_interval,
          max_cached_steps=args.max_cached_steps,
          max_continuous_cached_steps=args.max_continuous_cached_steps,
          residual_diff_threshold=args.residual_diff_threshold,
          enable_separate_cfg=kwargs.get("enable_separate_cfg", None),
          steps_computation_mask=steps_computation_mask,
          force_refresh_step_hint=kwargs.get("force_refresh_step_hint", None),
          force_refresh_step_policy=kwargs.get("force_refresh_step_policy", "once"),
          enable_seacache=args.seacache,
          seacache_mode=args.seacache_mode,
          seacache_feature_shape=args.seacache_feature_shape,
          seacache_power_exp=args.seacache_power_exp,
          enable_softcache=args.softcache,
          softcache_tau=args.softcache_tau,
          softcache_jf_max=args.softcache_jf_max,
        ) if cache_config is None and args.cache else cache_config),
        calibrator_config=build_calibrator_config(args),
        params_modifiers=kwargs.get("params_modifiers", None),
        parallelism_config=(ParallelismConfig(
          ulysses_size=ulysses_size,
          ring_size=ring_size,
          tp_size=tp_size,
          local_weight_shards=args.local_weight_shards,
          backend=ParallelismBackend.AUTO,
          attention_backend=("native" if not args.attn else args.attn),
          extra_parallel_modules=extra_parallel_modules,
          ulysses_anything=args.ulysses_anything,
          quant_communicate_kwargs=args.quant_communicate_params or {},
          ulysses_async=args.ulysses_async,
          ring_rotate_method=args.ring_rotate_method,
          ring_convert_to_fp32=not args.ring_no_convert_to_fp32,
        ) if parallelism_config is None and args.parallel_type is not None else parallelism_config),
        # Allow attention backend for non-parallelism case
        attention_backend=(("native" if not args.attn else args.attn)
                           if args.parallel_type is None else None),
      )
    else:
      # Apply acceleration configs from config path
      configs = load_configs(args.config_path)
      quantize_config = configs.get("quantize_config", None)
      if quantize_config is not None:
        args.quantize = True
        try:
          args.quantize_type = quantize_config.component_quant_types()["transformer"]
        except Exception:
          args.quantize_type = list(quantize_config.component_quant_types().values())[0]
        logger.info(f"Quantization config from {args.config_path}: {quantize_config.strify()}")

      enable_cache(
        pipe_or_adapter,
        **configs,
      )
      logger.info(f"Applied acceleration config from file: {args.config_path}.")
  else:
    logger.info("No caching or parallelism is applied.")
    if args.attn is not None:
      logger.info(f"Applying custom attention backend: {args.attn} ...")
      set_attn_backend(pipe_or_adapter, attention_backend=args.attn)

  # Quantization
  # WARN: Must apply quantization after tensor parallelism is applied.
  # torchao is compatible with tensor parallelism but requires to be
  # applied after TP.
  # Avoid quantization if quant_config is already applied via config file.
  if quantize_config is None:
    maybe_quantize_transformer(args, pipe_or_adapter)
    maybe_quantize_text_encoder(args, pipe_or_adapter)

  # VAE Tiling or Slicing
  maybe_vae_tiling_or_slicing(args, pipe_or_adapter)

  # Compilation
  maybe_compile_transformer(args, pipe_or_adapter)
  maybe_compile_text_encoder(args, pipe_or_adapter)
  maybe_compile_controlnet(args, pipe_or_adapter)
  maybe_compile_vae(args, pipe_or_adapter)

  if isinstance(pipe_or_adapter, BlockAdapter):
    pipe = pipe_or_adapter.pipe
  else:
    pipe = pipe_or_adapter
  should_skip_initial_pipeline_move = any(
    _should_skip_initial_few_shot_pipeline_move(getattr(pipe, name, None))
    for name in ("transformer", "transformer_2")) if pipe is not None else False
  _register_deferred_few_shot_pipeline_move_callbacks(args, pipe)

  # CPU Offload
  _, device = get_rank_device()
  used_adaln_host_residency = maybe_adaln_host_residency(args, pipe_or_adapter)
  used_diffusers_cpu_offload = maybe_cpu_offload(args, pipe_or_adapter)
  used_generic_module_offload = maybe_generic_module_offload(args, pipe_or_adapter)
  if (not used_adaln_host_residency and not used_diffusers_cpu_offload
      and not used_generic_module_offload):
    # Set device if no cpu offload
    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
    else:
      pipe = pipe_or_adapter
    if pipe is not None and not args.device_map_balance:
      if should_skip_initial_pipeline_move:
        logger.info(
          "Skipping eager pipe.to(%s) because SVDQ few-shot quantization will finish "
          "\nduring warmup or the current run before the low-memory layerwise pipeline move.",
          device,
        )
      else:
        pipe.to(device)

  return pipe_or_adapter
