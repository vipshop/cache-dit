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
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--alter", action="store_true", default=False)
    parser.add_argument("--l1-diff", action="store_true", default=False)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=1)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--Bn-steps", "--BnS", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-cached-steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
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
            "max_pruned_steps": args.max_cached_steps,  # -1 means no limit
            # releative token diff threshold, default is 0.0
            "important_condition_threshold": 0.00,
            "enable_dynamic_prune_threshold": True,
            "max_dynamic_prune_threshold": 2 * args.rdt,
            "dynamic_prune_threshold_relax_ratio": 1.25,
            "residual_cache_update_interval": 1,
            # You can set non-prune blocks to avoid ageressive pruning.
            # FLUX.1 has 19 + 38 blocks, so we can set it to 0, 2, 4, ..., etc.
            "non_prune_blocks_ids": CacheType.range(0, 19 + 38, 4),
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
    pruned_blocks = []
    actual_blocks = []
    pruned_steps = 0
    pruned_ratio = 0.0
    for i in range(args.repeats):
        start = time.time()
        image = pipe(
            "A cat holding a sign that says hello world with complex background",
            num_inference_steps=args.steps,
            generator=torch.Generator("cuda").manual_seed(args.seed),
        ).images[0]
        end = time.time()
        all_times.append(end - start)
        if hasattr(pipe.transformer, "_cached_steps"):
            cached_stepes = len(pipe.transformer._cached_steps)
        if hasattr(pipe.transformer, "_pruned_blocks"):
            pruned_blocks = pipe.transformer._pruned_blocks
        if hasattr(pipe.transformer, "_actual_blocks"):
            actual_blocks = pipe.transformer._actual_blocks
        if hasattr(pipe.transformer, "_pruned_steps"):
            pruned_steps = pipe.transformer._pruned_steps

        pruned_ratio = (
            sum(pruned_blocks) / sum(actual_blocks) if actual_blocks else 0
        ) * 100
        logger.info(
            f"Run {i + 1}/{args.repeats}, "
            f"Time: {all_times[-1]:.2f}s, "
            f"Cached Steps: {cached_stepes}, "
            f"Pruned Blocks: {sum(pruned_blocks)}({pruned_ratio:.2f})%, "
            f"Pruned Steps: {pruned_steps}"
        )
        if len(actual_blocks) > 0:
            logger.info(
                f"Actual Blocks: {actual_blocks}\n"
                f"Pruned Blocks: {pruned_blocks}"
            )

    all_times.pop(0)  # Remove the first run time, usually warmup
    mean_time = sum(all_times) / len(all_times)
    logger.info(
        f"Mean Time: {mean_time:.2f}s, "
        f"Cached Steps: {cached_stepes}, "
        f"Pruned Blocks: {sum(pruned_blocks)}({pruned_ratio:.2f})%, "
        f"Pruned Steps: {pruned_steps}"
    )
    if len(actual_blocks) > 0:
        logger.info(
            f"Actual Blocks: {actual_blocks}\n"
            f"Pruned Blocks: {pruned_blocks}"
        )
    if len(actual_blocks) > 0:
        save_name = (
            f"{cache_type}_R{args.rdt}_P{pruned_ratio:.1f}%_"
            f"T{mean_time:.2f}s.png"
        )
    else:
        save_name = (
            f"{cache_type}_R{args.rdt}_S{cached_stepes}%_"
            f"T{mean_time:.2f}s.png"
        )
    image.save(save_name)
    logger.info(f"Image saved as {save_name}")


if __name__ == "__main__":
    main()
