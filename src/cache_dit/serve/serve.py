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
        "--pipeline-quant-config-path",
        type=str,
        default=None,
        help="Path to custom Python module that provides get_pipeline_quant_config() function",
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
    # TODO: vae TP will be supported in the future
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
        "--attn",
        type=str,
        default=None,
        choices=[
            None,
            "flash",
            "_flash_3",
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


def maybe_init_distributed(args):
    import torch.distributed as dist

    if args.parallel_type is not None:
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl" if args.ulysses_anything else "nccl",
        )
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        torch.cuda.set_device(device)
        return rank, device
    return 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def launch_server(args=None):
    """Launch the serving server."""
    if args is None:
        args = parse_args()

    rank, device = maybe_init_distributed(args)
    if args.parallel_type is not None:
        import torch.distributed as dist

        logger.info(f"Initialized distributed: rank={rank}, world_size={dist.get_world_size()}")

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
    if args.parallel_type in ["ulysses", "ring"]:
        if hasattr(args, "attn") and args.attn is not None:
            parallel_args["attention_backend"] = args.attn
        else:
            parallel_args["attention_backend"] = "native"
        if hasattr(args, "ulysses_anything") and args.ulysses_anything:
            parallel_args["experimental_ulysses_anything"] = True
        if hasattr(args, "ulysses_float8") and args.ulysses_float8:
            parallel_args["experimental_ulysses_float8"] = True
        if hasattr(args, "ulysses_async") and args.ulysses_async:
            parallel_args["experimental_ulysses_async"] = True
    elif args.parallel_type == "tp":
        pass

    parallel_args["parallel_text_encoder"] = args.parallel_text_encoder
    parallel_args["parallel_vae"] = args.parallel_vae

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
        attn_backend=args.attn,
        quantize=args.quantize,
        quantize_type=args.quantize_type,
        pipeline_quant_config_path=args.pipeline_quant_config_path,
    )

    logger.info("Loading model...")
    model_manager.load_model()
    logger.info("Model loaded successfully!")

    # For TP and CP, we need all ranks to participate in inference
    # We use a simple broadcast mechanism to synchronize requests
    if args.parallel_type in ["tp", "ulysses", "ring"]:
        import torch.distributed as dist

        dist.barrier()
        logger.info(f"Rank {rank}: All ranks ready, starting service...")

        if rank == 0:
            # Rank 0: Start HTTP server and broadcast requests to other ranks
            from cache_dit.serve.tp_worker import TPCoordinator

            coordinator = TPCoordinator(model_manager, rank, dist.get_world_size())
            app = create_app(coordinator)

            logger.info(
                f"Starting distributed server (rank 0, {args.parallel_type}) at http://{args.host}:{args.port}"
            )
            logger.info(f"API docs at http://{args.host}:{args.port}/docs")

            uvicorn.run(
                app,
                host=args.host,
                port=args.port,
                workers=1,  # Must be 1 for distributed
                log_level="info",
            )
        else:
            # Other ranks: Run worker loop to receive and execute requests
            from cache_dit.serve.tp_worker import run_tp_worker

            logger.info(f"Starting distributed worker (rank {rank}, {args.parallel_type})")
            run_tp_worker(model_manager, rank)
    else:
        # Single GPU mode
        if rank == 0:
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
        else:
            # This should not happen in single GPU mode
            logger.warning(f"Rank {rank}: Unexpected rank in single GPU mode")
            import time

            while True:
                time.sleep(1)


if __name__ == "__main__":
    launch_server()
