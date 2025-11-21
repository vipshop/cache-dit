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
    parser.add_argument("--fuse-lora", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--Fn", type=int, default=8)
    parser.add_argument("--Bn", type=int, default=0)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--max-warmup-steps", "--w", type=int, default=8)
    parser.add_argument("--warmup-interval", "--wi", type=int, default=1)
    parser.add_argument("--max-cached-steps", "--mc", type=int, default=-1)
    parser.add_argument("--max-continuous-cached-steps", "--mcc", type=int, default=-1)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "-order", type=int, default=1)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--quantize", "-q", action="store_true", default=False)
    # float8, float8_weight_only, int8, int8_weight_only, int4, int4_weight_only
    parser.add_argument(
        "--quantize-type",
        type=str,
        default="float8_weight_only",
        choices=[
            "float8",
            "float8_weight_only",
            "int8",
            "int8_weight_only",
            "int4",
            "int4_weight_only",
            "bitsandbytes_4bit",
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
        "--attn",  # attention backend for context parallelism
        type=str,
        default=None,
        choices=[
            None,
            "flash",
            # Based on this fix: https://github.com/huggingface/diffusers/pull/12563
            "native",  # native pytorch attention: sdpa
            "_native_cudnn",
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
    parser.add_argument(
        "--track-memory",
        action="store_true",
        default=False,
        help="Track and report peak GPU memory usage",
    )
    parser.add_argument(
        "--ulysses-anything",
        action="store_true",
        default=False,
        help="Enable Ulysses Anything Attention for context parallelism",
    )
    return parser.parse_args() if parse else parser


def cachify(
    args,
    pipe_or_adapter,
    **kwargs,
):
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
                "attention_backend": ("_native_cudnn" if not args.attn else args.attn),
                "experimental_ulysses_anything": args.ulysses_anything,
            }
            if backend == ParallelismBackend.NATIVE_DIFFUSER
            else None
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
    quantize_type = args.quantize_type if args.quantize else ""
    if quantize_type != "":
        quantize_type = f"_{quantize_type}"
    base_str = (
        f"C{int(args.compile)}_Q{int(args.quantize)}{quantize_type}_"
        f"{cache_dit.strify(pipe_or_stats)}"
    )
    if args.ulysses_anything:
        base_str += "_ulysses_anything"
    return base_str


def maybe_init_distributed(args=None):
    if args is not None:
        if args.parallel_type is not None:
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl",
            )
            rank = dist.get_rank()
            device = torch.device("cuda", rank % torch.cuda.device_count())
            torch.cuda.set_device(device)
            return rank, device
    else:
        # always init distributed for other examples
        if not dist.is_initialized():
            dist.init_process_group(
                backend="cpu:gloo,cuda:nccl",
            )
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
        return rank, device
    return 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_destroy_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()
