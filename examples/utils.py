import argparse

import torch
import torch.distributed as dist

import cache_dit
from cache_dit import init_logger
from cache_dit.parallelism.parallel_backend import ParallelismBackend

logger = init_logger(__name__)


class MemoryTracker:
    """Track peak GPU memory usage during execution."""

    def __init__(self, device=None):
        self.device = device if device is not None else torch.cuda.current_device()
        self.enabled = torch.cuda.is_available()
        self.peak_memory = 0

    def __enter__(self):
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            torch.cuda.synchronize(self.device)
            self.peak_memory = torch.cuda.max_memory_allocated(self.device)

    def get_peak_memory_gb(self):
        """Get peak memory in GB."""
        return self.peak_memory / (1024**3)

    def report(self):
        """Print memory usage report."""
        if self.enabled:
            peak_gb = self.get_peak_memory_gb()
            logger.info(f"Peak GPU memory usage: {peak_gb:.2f} GB")
            return peak_gb
        return 0


def GiB():
    try:
        if not torch.cuda.is_available():
            return 0
        total_memory_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device(),
        ).total_memory
        total_memory_gib = total_memory_bytes / (1024**3)
        return int(total_memory_gib)
    except Exception:
        return 0


def get_args(
    parse: bool = True,
) -> argparse.ArgumentParser | argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--compile-repeated-blocks", action="store_true", default=False)
    parser.add_argument("--compile-vae", action="store_true", default=False)
    parser.add_argument("--compile-text-encoder", action="store_true", default=False)
    parser.add_argument("--max-autotune", action="store_true", default=False)
    parser.add_argument("--fuse-lora", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--Fn", type=int, default=8)
    parser.add_argument("--Bn", type=int, default=0)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--max-warmup-steps", "--wa", type=int, default=8)
    parser.add_argument("--warmup-interval", "--wi", type=int, default=1)
    parser.add_argument("--max-cached-steps", "--mc", type=int, default=-1)
    parser.add_argument("--max-continuous-cached-steps", "--mcc", type=int, default=-1)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "-order", type=int, default=1)
    parser.add_argument("--steps-mask", "--scm", action="store_true", default=False)
    parser.add_argument(
        "--mask-policy",
        type=str,
        default=None,
        choices=[
            None,
            "slow",
            "medium",
            "fast",
            "ultra",
        ],
        help="Pre-defined steps computation mask policy",
    )
    parser.add_argument("--height", "--h", type=int, default=None)
    parser.add_argument("--width", "--w", type=int, default=None)
    parser.add_argument("--quantize", "--q", action="store_true", default=False)
    # float8, float8_weight_only, int8, int8_weight_only, int4, int4_weight_only
    parser.add_argument(
        "--quantize-type",
        "--q-type",
        type=str,
        default=None,
        choices=[
            None,
            "float8",
            "float8_weight_only",
            "float8_wo",  # alias for float8_weight_only
            "int8",
            "int8_weight_only",
            "int8_wo",  # alias for int8_weight_only
            "int4",
            "int4_weight_only",
            "int4_wo",  # alias for int4_weight_only
            "bitsandbytes_4bit",
            "bnb_4bit",  # alias for bitsandbytes_4bit
        ],
    )
    parser.add_argument(
        "--parallel-type",
        "--parallel",
        type=str,
        default=None,
        choices=[
            None,
            "tp",
            "ulysses",
            "ring",
        ],
    )
    parser.add_argument(
        "--parallel-vae",
        action="store_true",
        default=False,
        help="Enable VAE parallelism if applicable.",
    )
    parser.add_argument(
        "--parallel-text-encoder",
        "--parallel-text",
        action="store_true",
        default=False,
        help="Enable text encoder parallelism if applicable.",
    )
    parser.add_argument(
        "--attn",  # attention backend for context parallelism
        type=str,
        default=None,
        choices=[
            None,
            "flash",
            "_flash_3",  # FlashAttention-3
            # Based on this fix: https://github.com/huggingface/diffusers/pull/12563
            "native",  # native pytorch attention: sdpa
            "_native_cudnn",
            # '_sdpa_cudnn' is only in cache-dit to support context parallelism
            # with attn masks, e.g., ZImage. It is not in diffusers yet.
            "_sdpa_cudnn",
            "sage",  # Need install sageattention: https://github.com/thu-ml/SageAttention
        ],
    )
    parser.add_argument("--perf", action="store_true", default=False)
    # New arguments for customization
    parser.add_argument("--prompt", type=str, default=None, help="Override default prompt")
    parser.add_argument(
        "--negative-prompt", type=str, default=None, help="Override default negative prompt"
    )
    parser.add_argument("--model-path", type=str, default=None, help="Override model path")
    parser.add_argument("--image-path", type=str, default=None, help="Override image path")
    parser.add_argument(
        "--track-memory",
        action="store_true",
        default=False,
        help="Track and report peak GPU memory usage",
    )
    parser.add_argument(
        "--ulysses-anything",
        "--uaa",
        action="store_true",
        default=False,
        help="Enable Ulysses Anything Attention for context parallelism",
    )
    parser.add_argument(
        "--ulysses-float8",
        "--ufp8",
        action="store_true",
        default=False,
        help="Enable Ulysses Attention/UAA Float8 for context parallelism",
    )
    parser.add_argument(
        "--ulysses-async",
        "--uaqkv",
        action="store_true",
        default=False,
        help="Enabled experimental Async QKV Projection with Ulysses for context parallelism",
    )
    parser.add_argument(
        "--disable-compute-comm-overlap",
        "--dcco",
        action="store_true",
        default=False,
        help="Disable compute-communication overlap during compilation",
    )
    parser.add_argument("--profile", action="store_true", default=False)
    parser.add_argument("--profile-name", type=str, default=None)
    parser.add_argument("--profile-dir", type=str, default=None)
    parser.add_argument(
        "--profile-activities",
        type=str,
        nargs="+",
        default=["CPU", "GPU"],
        choices=["CPU", "GPU", "MEM"],
    )
    parser.add_argument("--profile-with-stack", action="store_true", default=True)
    parser.add_argument("--profile-record-shapes", action="store_true", default=True)
    args_or_parser = parser.parse_args() if parse else parser
    if parse:
        if args_or_parser.quantize_type is not None:
            # Force enable quantization if quantize_type is specified
            args_or_parser.quantize = True
        if args_or_parser.quantize and args_or_parser.quantize_type is None:
            args_or_parser.quantize_type = "float8_weight_only"
        # Handle alias for quantize_type
        if args_or_parser.quantize_type == "float8_wo":  # alias
            args_or_parser.quantize_type = "float8_weight_only"
        if args_or_parser.quantize_type == "int8_wo":  # alias
            args_or_parser.quantize_type = "int8_weight_only"
        if args_or_parser.quantize_type == "int4_wo":  # alias
            args_or_parser.quantize_type = "int4_weight_only"
        if args_or_parser.quantize_type == "bnb_4bit":  # alias
            args_or_parser.quantize_type = "bitsandbytes_4bit"
    return args_or_parser


def cachify(
    args,
    pipe_or_adapter,
    **kwargs,
):
    if args.disable_compute_comm_overlap:
        # Enable compute comm overlap default for torch.compile if used
        # cache_dit.set_compile_flags(), users need to disable it explicitly.
        cache_dit.disable_compute_comm_overlap()

    if args.cache or args.parallel_type is not None:
        import torch.distributed as dist

        from cache_dit import DBCacheConfig, ParallelismConfig, TaylorSeerCalibratorConfig

        cache_config = kwargs.pop("cache_config", None)
        parallelism_config = kwargs.pop("parallelism_config", None)

        backend = (
            ParallelismBackend.NATIVE_PYTORCH
            if args.parallel_type in ["tp"]
            else ParallelismBackend.NATIVE_DIFFUSER
        )

        parallel_kwargs = (
            {
                "attention_backend": ("native" if not args.attn else args.attn),
                "experimental_ulysses_anything": args.ulysses_anything,
                "experimental_ulysses_float8": args.ulysses_float8,
                "experimental_ulysses_async": args.ulysses_async,
                # e.g., text_encoder_2 in FluxPipeline, text_encoder in Flux2Pipeline
                "extra_parallel_modules": kwargs.get("extra_parallel_modules", []),
            }
            if backend == ParallelismBackend.NATIVE_DIFFUSER
            else {
                # e.g., text_encoder_2 in FluxPipeline, text_encoder in Flux2Pipeline
                "extra_parallel_modules": kwargs.get("extra_parallel_modules", []),
            }
        )
        cache_dit.enable_cache(
            pipe_or_adapter,
            cache_config=(
                DBCacheConfig(
                    Fn_compute_blocks=args.Fn,
                    Bn_compute_blocks=args.Bn,
                    max_warmup_steps=args.max_warmup_steps,
                    warmup_interval=args.warmup_interval,
                    max_cached_steps=args.max_cached_steps,
                    max_continuous_cached_steps=args.max_continuous_cached_steps,
                    residual_diff_threshold=args.rdt,
                    enable_separate_cfg=kwargs.get("enable_separate_cfg", None),
                    steps_computation_mask=kwargs.get("steps_computation_mask", None),
                )
                if cache_config is None and args.cache
                else cache_config
            ),
            calibrator_config=(
                TaylorSeerCalibratorConfig(
                    taylorseer_order=args.taylorseer_order,
                )
                if args.taylorseer
                else None
            ),
            params_modifiers=kwargs.get("params_modifiers", None),
            parallelism_config=(
                ParallelismConfig(
                    ulysses_size=(
                        dist.get_world_size() if args.parallel_type == "ulysses" else None
                    ),
                    ring_size=(dist.get_world_size() if args.parallel_type == "ring" else None),
                    tp_size=(dist.get_world_size() if args.parallel_type == "tp" else None),
                    backend=backend,
                    parallel_kwargs=parallel_kwargs,
                )
                if parallelism_config is None and args.parallel_type in ["ulysses", "ring", "tp"]
                else parallelism_config
            ),
        )

    return pipe_or_adapter


def strify(args, pipe_or_stats):
    base_str = ""
    if args.height is not None and args.width is not None:
        base_str += f"{args.height}x{args.width}_"
    quantize_type = args.quantize_type if args.quantize else ""
    if quantize_type != "":
        quantize_type = f"_{quantize_type}"
    base_str += (
        f"C{int(args.compile)}_Q{int(args.quantize)}{quantize_type}_"
        f"{cache_dit.strify(pipe_or_stats)}"
    )
    if args.ulysses_anything:
        base_str += "_ulysses_anything"
        if args.ulysses_float8:
            base_str += "_float8"
    else:
        if args.ulysses_float8:
            base_str += "_ulysses_float8"
    if args.ulysses_async:
        base_str += "_ulysses_async"
    if args.attn is not None:
        base_str += f"_{args.attn.strip('_')}"
    return base_str


def maybe_init_distributed(args=None):
    if args is not None:
        if args.parallel_type is not None:
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl" if args.ulysses_anything else "nccl",
            )
            rank = dist.get_rank()
            device = torch.device("cuda", rank % torch.cuda.device_count())
            torch.cuda.set_device(device)
            return rank, device
    else:
        # always init distributed for other examples
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
            )
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
        return rank, device
    return 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_destroy_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def create_profiler_from_args(args, profile_name=None):
    from cache_dit.profiler import ProfilerContext

    return ProfilerContext(
        enabled=args.profile,
        activities=getattr(args, "profile_activities", ["CPU", "GPU"]),
        output_dir=getattr(args, "profile_dir", None),
        profile_name=profile_name or getattr(args, "profile_name", None),
        with_stack=getattr(args, "profile_with_stack", True),
        record_shapes=getattr(args, "profile_record_shapes", True),
    )
