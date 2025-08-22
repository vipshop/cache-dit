import os
import argparse
import torch
import random
import time
from diffusers import FluxPipeline, FluxTransformer2DModel

import cache_dit

logger = cache_dit.init_logger(__name__)


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--cache-config", type=str, default=None)
    parser.add_argument("--alter", action="store_true", default=False)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=2)
    parser.add_argument("--l1-diff", action="store_true", default=False)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=1)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-cached-steps", type=int, default=-1)
    parser.add_argument("--gen-device", type=str, default="cpu")
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--inductor-flags", action="store_true", default=False)
    parser.add_argument("--compile-all", action="store_true", default=False)
    parser.add_argument(
        "--unified-api", "--uapi", action="store_true", default=False
    )
    return parser.parse_args()


def set_rand_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def get_cache_options(cache_type, args: argparse.Namespace = None):
    assert args is not None
    if args.cache_config is not None:
        assert os.path.exists(args.cache_config)

        cache_options = cache_dit.load_options(args.cache_config)
        logger.info(
            f"Loaded cache options from file: {args.cache_config}, "
            f"\n{cache_options}"
        )
        if cache_type is None:
            cache_type = cache_options["cache_type"]
    else:
        cache_type = cache_dit.cache_type(cache_type)
        if cache_type == cache_dit.DBCache:
            cache_options = {
                "cache_type": cache_dit.DBCache,
                "warmup_steps": args.warmup_steps,
                "max_cached_steps": args.max_cached_steps,  # -1 means no limit
                "Fn_compute_blocks": args.Fn_compute_blocks,  # Fn, F8, etc.
                "Bn_compute_blocks": args.Bn_compute_blocks,  # Bn, B16, etc.
                "max_Fn_compute_blocks": 19,
                "max_Bn_compute_blocks": 38,
                "non_compute_blocks_diff_threshold": 0.08,
                "residual_diff_threshold": args.rdt,
                # Alter cache pattern: 0 F 1 T 2 F 3 T 4 F 5 T ...
                "enable_alter_cache": args.alter,
                "alter_residual_diff_threshold": args.rdt,
                "l1_hidden_states_diff_threshold": (
                    None if not args.l1_diff else args.rdt
                ),
                # releative token diff threshold, default is 0.0
                "important_condition_threshold": 0.00,
                # TaylorSeer options
                "enable_taylorseer": args.taylorseer,
                "enable_encoder_taylorseer": args.taylorseer,
                # Taylorseer cache type cache be hidden_states or residual
                "taylorseer_cache_type": "residual",
                "taylorseer_kwargs": {
                    "n_derivatives": args.taylorseer_order,
                },
            }
        else:
            cache_options = {
                "cache_type": cache_dit.NONE,
            }
    # Reset cache_type for result saving
    cache_type_str = str(cache_type).removeprefix("CacheType.").upper()
    if cache_type == cache_dit.DBCache:
        cache_type_str = (
            f"{cache_type_str}_F{args.Fn_compute_blocks}"
            f"B{args.Bn_compute_blocks}W{args.warmup_steps}"
            f"T{int(args.taylorseer)}O{args.taylorseer_order}"
        )
    return cache_options, cache_type_str


@torch.no_grad()
def main():
    args = get_args()
    logger.info(f"Arguments: {args}")
    set_rand_seeds(args.seed)

    cache_options, cache_type = get_cache_options(args.cache, args)

    logger.info(f"Cache Type: {cache_type}")
    logger.info(f"Cache Options: {cache_options}")

    pipe = FluxPipeline.from_pretrained(
        os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Apply cache to the pipeline
    if not args.unified_api:
        cache_dit.enable_cache(pipe, **cache_options)
    else:
        assert isinstance(pipe.transformer, FluxTransformer2DModel)
        from cache_dit import ForwardPattern, BlockAdapter

        cache_dit.enable_cache(
            BlockAdapter(
                pipe,
                transformer=pipe.transformer,
                blocks=(
                    pipe.transformer.transformer_blocks
                    + pipe.transformer.single_transformer_blocks
                ),
                blocks_name="transformer_blocks",
                dummy_blocks_names=["single_transformer_blocks"],
            ),
            forward_pattern=ForwardPattern.Pattern_1,
            **cache_options,
        )

    if args.compile:
        # Increase recompile limit for DBCache
        if args.inductor_flags:
            cache_dit.set_compile_configs()
        else:
            torch._dynamo.config.recompile_limit = 96  # default is 8
            torch._dynamo.config.accumulated_recompile_limit = (
                2048  # default is 256
            )
        if isinstance(pipe.transformer, FluxTransformer2DModel):
            if not args.compile_all:
                logger.warning(
                    "Only compile transformer blocks not the whole model "
                    "for FluxTransformer2DModel to keep higher precision."
                )
                for module in pipe.transformer.transformer_blocks:
                    module.compile(fullgraph=True)
                for module in pipe.transformer.single_transformer_blocks:
                    module.compile(fullgraph=True)
            else:
                pipe.transformer = torch.compile(
                    pipe.transformer, mode="default"
                )
        else:
            logger.info("Compiling the transformer with default mode.")
            pipe.transformer = torch.compile(pipe.transformer, mode="default")

    all_times = []
    for i in range(args.repeats):
        start = time.time()
        image = pipe(
            "A cat holding a sign that says hello world",
            num_inference_steps=args.steps,
            generator=torch.Generator(args.gen_device).manual_seed(args.seed),
        ).images[0]
        end = time.time()
        all_times.append(end - start)
        logger.info(
            f"Run {i + 1}/{args.repeats}, " f"Time: {all_times[-1]:.2f}s"
        )

    all_times.pop(0)  # Remove the first run time, usually warmup
    mean_time = sum(all_times) / len(all_times)
    logger.info(f"Mean Time: {mean_time:.2f}s")

    stats = cache_dit.summary(pipe)
    save_name = (
        f"C{int(args.compile)}_{cache_type}_"
        f"R{args.rdt}_S{len(stats.cached_steps)}_"
        f"T{mean_time:.2f}s.png"
    )

    image.save(save_name)
    logger.info(f"Image saved as {save_name}")


if __name__ == "__main__":
    main()
