"""Server launcher for cache-dit.

Adapted from SGLang's server launcher:
https://github.com/sgl-project/sglang/blob/main/python/sglang/launch_server.py
"""

import argparse
import torch
import uvicorn
from cache_dit.serve.model_manager import ModelManager
from cache_dit.serve.api_server import create_app
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def get_args(
    parse: bool = True,
) -> argparse.ArgumentParser | argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--compile-repeated-blocks", action="store_true", default=False)
    parser.add_argument("--max-autotune", action="store_true", default=False)
    parser.add_argument("--fuse-lora", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--repeat", type=int, default=None)
    parser.add_argument("--Fn", type=int, default=8)
    parser.add_argument("--Bn", type=int, default=0)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--max-warmup-steps", "--w", type=int, default=8)
    parser.add_argument("--warmup-interval", "--wi", type=int, default=1)
    parser.add_argument("--max-cached-steps", "--mc", type=int, default=-1)
    parser.add_argument("--max-continuous-cached-steps", "--mcc", type=int, default=-1)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "-order", type=int, default=1)
    parser.add_argument("--steps-mask", "--scm", action="store_true", default=False)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--quantize", "-q", action="store_true", default=False)
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
            "bnb_4bit",
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
        "--attn",
        type=str,
        default=None,
        choices=[
            None,
            "flash",
            "native",
            "_native_cudnn",
            "_sdpa_cudnn",
            "sage",
        ],
    )
    parser.add_argument("--perf", action="store_true", default=False)
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
        "--ulysses-async-qkv-proj",
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
        if args_or_parser.quantize_type == "bnb_4bit":
            args_or_parser.quantize_type = "bitsandbytes_4bit"
    return args_or_parser


def parse_args():
    parser = get_args(parse=False)

    # Add server-specific arguments
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu), auto-detect by default",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        default=False,
        help="Enable CPU offload (saves GPU memory)",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        help="Device map strategy (e.g., balanced)",
    )

    args = parser.parse_args()

    # Handle quantize_type alias
    if hasattr(args, "quantize_type") and args.quantize_type == "bnb_4bit":
        args.quantize_type = "bitsandbytes_4bit"

    # Ensure model_path is required
    if not args.model_path:
        parser.error("--model-path is required")

    return args


def launch_server(args=None):
    """Launch the serving server."""
    if args is None:
        args = parse_args()

    torch_dtype = getattr(torch, args.dtype)

    # Use cache argument from utils.get_args
    enable_cache = args.cache
    cache_config = None
    if enable_cache:
        cache_config = {
            "residual_diff_threshold": args.rdt,
            "Fn_compute_blocks": args.Fn,
            "Bn_compute_blocks": args.Bn,
            "max_warmup_steps": args.max_warmup_steps,
            "warmup_interval": args.warmup_interval,
            "max_cached_steps": args.max_cached_steps,
            "max_continuous_cached_steps": args.max_continuous_cached_steps,
        }

    parallel_args = {}
    if hasattr(args, "attn") and args.attn is not None:
        parallel_args["attn_backend"] = args.attn
    if args.parallel_type is not None:
        if hasattr(args, "ulysses_anything") and args.ulysses_anything:
            parallel_args["ulysses_anything"] = True
        if hasattr(args, "ulysses_float8") and args.ulysses_float8:
            parallel_args["ulysses_float8"] = True
        if hasattr(args, "ulysses_async_qkv_proj") and args.ulysses_async_qkv_proj:
            parallel_args["ulysses_async_qkv_proj"] = True

    logger.info("Initializing model manager...")
    model_manager = ModelManager(
        model_path=args.model_path,
        device=args.device or "cuda",
        torch_dtype=torch_dtype,
        enable_cache=enable_cache,
        cache_config=cache_config,
        enable_cpu_offload=args.enable_cpu_offload,
        device_map=args.device_map,
        enable_compile=args.compile,
        parallel_type=args.parallel_type,
        parallel_args=parallel_args,
    )

    logger.info("Loading model...")
    model_manager.load_model()
    logger.info("Model loaded successfully!")

    app = create_app(model_manager)

    logger.info(f"Starting server at http://{args.host}:{args.port}")
    logger.info(f"API docs at http://{args.host}:{args.port}/docs")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    launch_server()
