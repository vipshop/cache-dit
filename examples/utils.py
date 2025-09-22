import torch
import argparse
import cache_dit
from cache_dit import init_logger

logger = init_logger(__name__)


def GiB():
    if not torch.cuda.is_available():
        return 0

    try:
        total_memory_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device(),
        ).total_memory
        total_memory_gib = total_memory_bytes / (1024**3)
        return int(total_memory_gib)
    except Exception:
        return 0


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--fuse-lora", action="store_true", default=False)
    parser.add_argument("--quantize", "-q", action="store_true", default=False)
    parser.add_argument(
        "--quantize-type",
        "-type",
        type=str,
        default="fp8_w8a8_dq",
        choices=[
            "fp8_w8a8_dq",
            "fp8_w8a16_wo",
            "int8_w8a8_dq",
            "int8_w8a16_wo",
            "int4_w4a8_dq",
            "int4_w4a4_dq",
            "int4_w4a16_wo",
        ],
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--Fn", type=int, default=8)
    parser.add_argument("--Bn", type=int, default=0)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--max-warmup-steps", "-w", type=int, default=8)
    parser.add_argument("--max-cached-steps", "-mc", type=int, default=-1)
    parser.add_argument(
        "--max-continuous-cached-steps", "-mcc", type=int, default=-1
    )
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "-order", type=int, default=1)
    return parser.parse_args()


def cachify(
    args,
    pipe_or_adapter,
    **kwargs,
):
    if args.cache:
        from cache_dit import BasicCacheConfig, TaylorSeerCalibratorConfig

        specific_cache_config = kwargs.pop("cache_config", None)

        cache_dit.enable_cache(
            pipe_or_adapter,
            cache_config=(
                BasicCacheConfig(
                    Fn_compute_blocks=args.Fn,
                    Bn_compute_blocks=args.Bn,
                    max_warmup_steps=args.max_warmup_steps,
                    max_cached_steps=args.max_cached_steps,
                    max_continuous_cached_steps=args.max_continuous_cached_steps,
                    residual_diff_threshold=args.rdt,
                    **kwargs,
                )
                if specific_cache_config is None
                else specific_cache_config
            ),
            calibrator_config=(
                TaylorSeerCalibratorConfig(
                    taylorseer_order=args.taylorseer_order,
                )
                if args.taylorseer
                else None
            ),
        )

    return pipe_or_adapter


def strify(args, pipe_or_stats):
    return (
        f"C{int(args.compile)}_L{int(args.fuse_lora)}_Q{int(args.quantize)}"
        f"{'' if not args.quantize else ('_' + args.quantize_type)}_"
        f"{cache_dit.strify(pipe_or_stats)}"
    )
