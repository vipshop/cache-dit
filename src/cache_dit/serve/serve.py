"""Server launcher for cache-dit.

Adapted from SGLang's server launcher:
https://github.com/sgl-project/sglang/blob/main/python/sglang/launch_server.py
"""

import sys
import os

# Add examples directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../examples"))

import torch
import uvicorn
from utils import get_args
from cache_dit.serve.model_manager import ModelManager
from cache_dit.serve.api_server import create_app
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def parse_args():
    """Parse command line arguments using utils.get_args and add server-specific args."""
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
            "enable_separate_cfg": True,
        }

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
