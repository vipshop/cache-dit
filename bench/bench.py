import os
import argparse
import torch
import time

from diffusers import FluxPipeline
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType
from cache_dit.logger import init_logger


logger = init_logger(__name__)


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--alter", action="store_true", default=False)
    parser.add_argument("--l1-diff", action="store_true", default=False)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=1)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--Bn-steps", "--BnS", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-cached-steps", type=int, default=-1)
    return parser.parse_args()


def get_cache_options(cache_type: CacheType, args: argparse.Namespace):
    cache_type = CacheType.type(cache_type)
    if cache_type == CacheType.FBCache:
        cache_options = {
            "cache_type": CacheType.FBCache,
            "warmup_steps": args.warmup_steps,
            "max_cached_steps": args.max_cached_steps,
            "residual_diff_threshold": args.rdt,
        }
    elif cache_type == CacheType.DBCache:
        cache_options = {
            "cache_type": CacheType.DBCache,
            "warmup_steps": args.warmup_steps,
            "max_cached_steps": args.max_cached_steps,  # -1 means no limit
            # Fn=1, Bn=0, means FB Cache, otherwise, Dual Block Cache
            "Fn_compute_blocks": args.Fn_compute_blocks,  # Fn, F8, etc.
            "Bn_compute_blocks": args.Bn_compute_blocks,  # Bn, B16, etc.
            "max_Fn_compute_blocks": 19,
            "max_Bn_compute_blocks": 38,
            # WARN: DON'T set len(Fn_compute_blocks_ids) > 0 NOW, still have
            # some precision issues. 0, 1, 2, ..., 7, etc.
            "Fn_compute_blocks_ids": [],
            # NOTE: Only skip the specific Bn blocks in cache steps.
            # 0, 2, 4, ..., 14, 15, etc.
            "Bn_compute_blocks_ids": CacheType.range(
                0, args.Bn_compute_blocks, args.Bn_steps
            ),
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
        }
    elif cache_type == CacheType.DBPrune:
        cache_options = {
            "cache_type": CacheType.DBPrune,
            "residual_diff_threshold": args.rdt,
            "Fn_compute_blocks": args.Fn_compute_blocks,
            "Bn_compute_blocks": args.Bn_compute_blocks,
            "warmup_steps": args.warmup_steps,
            "max_pruned_steps": -1,  # -1 means no limit
            # releative token diff threshold, default is 0.0
            "important_condition_threshold": 0.00,
        }
    else:
        cache_options = {
            "cache_type": CacheType.NONE,
        }
    # Reset cache_type for result saving
    cache_type_str = str(cache_type).removeprefix("CacheType.").upper()
    if cache_type == CacheType.DBCache:
        cache_type_str = (
            f"{cache_type_str}_F{args.Fn_compute_blocks}"
            f"B{args.Bn_compute_blocks}S{args.Bn_steps}"
        )
    elif cache_type == CacheType.DBPrune:
        cache_type_str = (
            f"{cache_type_str}_F{args.Fn_compute_blocks}"
            f"B{args.Bn_compute_blocks}"
        )
    return cache_options, cache_type_str


def main():
    args = get_args()
    logger.info(f"Arguments: {args}")

    pipe = FluxPipeline.from_pretrained(
        os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    cache_options, cache_type = get_cache_options(args.cache, args)

    logger.info(f"Cache Type: {cache_type}")
    logger.info(f"Cache Options: {cache_options}")

    apply_cache_on_pipe(pipe, **cache_options)

    all_times = []
    cached_stepes = 0
    for i in range(args.repeats):
        start = time.time()
        image = pipe(
            "A cat holding a sign that says hello world with complex background",
            num_inference_steps=args.steps,
            generator=torch.Generator("cuda").manual_seed(0),
        ).images[0]
        end = time.time()
        all_times.append(end - start)
        if hasattr(pipe.transformer, "_cached_steps"):
            cached_stepes = len(pipe.transformer._cached_steps)
        logger.info(
            f"Run {i + 1}/{args.repeats}, Time: {all_times[-1]:.2f}s, "
            f"Cached Steps: {cached_stepes}"
        )

    all_times.pop(0)  # Remove the first run time, usually warmup
    mean_time = sum(all_times) / len(all_times)
    logger.info(f"Mean Time: {mean_time:.2f}s, Cached Steps: {cached_stepes}")
    save_name = f"{cache_type}_R{args.rdt}_S{cached_stepes}.png"
    image.save(save_name)
    logger.info(f"Image saved as {save_name}")


if __name__ == "__main__":
    main()
