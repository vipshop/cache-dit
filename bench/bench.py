import os
import argparse
import torch
import time

from diffusers import FluxPipeline, FluxTransformer2DModel
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType
from cache_dit.logger import init_logger


logger = init_logger(__name__)


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", type=str, default=None)
    parser.add_argument("--alter", action="store_true", default=False)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=2)
    parser.add_argument("--l1-diff", action="store_true", default=False)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=1)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--Bn-steps", "--BnS", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-cached-steps", type=int, default=-1)
    parser.add_argument("--max-pruned-steps", type=int, default=-1)
    parser.add_argument("--gen-device", type=str, default="cuda")
    parser.add_argument("--ulysses", type=int, default=None)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument(
        "--force-compile-all",
        "--compile-all",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def get_cache_options(cache_type: CacheType, args: argparse.Namespace):
    cache_type = CacheType.type(cache_type)
    if cache_type == CacheType.FBCache:
        cache_options = {
            "cache_type": CacheType.FBCache,
            "warmup_steps": args.warmup_steps,
            "max_cached_steps": args.max_cached_steps,
            "residual_diff_threshold": args.rdt,
            # TaylorSeer options
            "enable_taylorseer": args.taylorseer,
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
            # Skip the specific Bn blocks in cache steps.
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
            # TaylorSeer options
            "enable_taylorseer": args.taylorseer,
            "enable_encoder_taylorseer": args.taylorseer,
            # Taylorseer cache type cache be hidden_states or residual
            "taylorseer_cache_type": "residual",
            "taylorseer_kwargs": {
                "n_derivatives": args.taylorseer_order,
            },
        }
    elif cache_type == CacheType.DBPrune:
        assert (
            args.taylorseer is False
        ), "DBPrune does not support TaylorSeer yet."
        cache_options = {
            "cache_type": CacheType.DBPrune,
            "residual_diff_threshold": args.rdt,
            "Fn_compute_blocks": args.Fn_compute_blocks,
            "Bn_compute_blocks": args.Bn_compute_blocks,
            "warmup_steps": args.warmup_steps,
            "max_pruned_steps": args.max_pruned_steps,  # -1 means no limit
            # releative token diff threshold, default is 0.0
            "important_condition_threshold": 0.00,
            "enable_dynamic_prune_threshold": (
                True if args.rdt <= 0.15 else False
            ),
            "max_dynamic_prune_threshold": 2 * args.rdt,
            "dynamic_prune_threshold_relax_ratio": 1.25,
            "residual_cache_update_interval": 1,
            # You can set non-prune blocks to avoid ageressive pruning.
            # FLUX.1 has 19 + 38 blocks, so we can set it to 0, 2, 4, ..., etc.
            "non_prune_blocks_ids": [],
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
            f"W{args.warmup_steps}T{int(args.taylorseer)}"
            f"O{args.taylorseer_order}"
        )
    elif cache_type == CacheType.DBPrune:
        cache_type_str = (
            f"{cache_type_str}_F{args.Fn_compute_blocks}"
            f"B{args.Bn_compute_blocks}W{args.warmup_steps}"
        )
    elif cache_type == CacheType.FBCache:
        cache_type_str = (
            f"{cache_type_str}_W{args.warmup_steps}T{int(args.taylorseer)}"
        )
    return cache_options, cache_type_str


@torch.no_grad()
def main():
    args = get_args()
    logger.info(f"Arguments: {args}")

    # Context Parallel from ParaAttention
    if args.ulysses is not None:
        try:
            import torch.distributed as dist
            from para_attn.context_parallel import init_context_parallel_mesh
            from para_attn.context_parallel.diffusers_adapters import (
                parallelize_pipe,
            )

            # Initialize distributed process group
            dist.init_process_group()
            torch.cuda.set_device(dist.get_rank())

            logger.info(f"Ulysses: {args.ulysses}")

            pipe = FluxPipeline.from_pretrained(
                os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
                torch_dtype=torch.bfloat16,
            ).to("cuda")

            parallelize_pipe(
                pipe,
                mesh=init_context_parallel_mesh(
                    pipe.device.type, max_ulysses_dim_size=args.ulysses
                ),
            )
        except ImportError as e:
            logger.error(
                "para-attn is not installed, please install it "
                "with `pip install para-attn.`"
            )
            args.ulysses = None
            raise e
    else:
        pipe = FluxPipeline.from_pretrained(
            os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
            torch_dtype=torch.bfloat16,
        ).to("cuda")

    cache_options, cache_type = get_cache_options(args.cache, args)

    logger.info(f"Cache Type: {cache_type}")
    logger.info(f"Cache Options: {cache_options}")

    # Apply cache to the pipeline
    apply_cache_on_pipe(pipe, **cache_options)

    if args.compile:
        # Increase recompile limit for DBCache and DBPrune while
        # using dynamic input shape.
        torch._dynamo.config.recompile_limit = 96  # default is 8
        torch._dynamo.config.accumulated_recompile_limit = (
            2048  # default is 256
        )
        if isinstance(pipe.transformer, FluxTransformer2DModel):
            logger.warning(
                "Only compile transformer blocks not the whole model "
                "for FluxTransformer2DModel to keep higher precision."
            )
            if (
                args.taylorseer_order <= 2
                or not args.taylorseer
                or args.force_compile_all
            ):
                # NOTE: Seems like compiling the whole transformer
                # will cause precision issues while using TaylorSeer
                # with order > 2.
                for module in pipe.transformer.transformer_blocks:
                    module.compile()
            else:
                logger.warning(
                    "Compiling the whole transformer model with TaylorSeer "
                    "order > 2 may cause precision issues. Skipping "
                    "transformer_blocks."
                )
            for module in pipe.transformer.single_transformer_blocks:
                module.compile()
        else:
            logger.info("Compiling the transformer with default mode.")
            pipe.transformer = torch.compile(pipe.transformer, mode="default")

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
            generator=torch.Generator(args.gen_device).manual_seed(args.seed),
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
    ulysses = 0 if args.ulysses is None else args.ulysses
    if len(actual_blocks) > 0:
        save_name = (
            f"U{ulysses}_C{int(args.compile)}_{cache_type}_"
            f"R{args.rdt}_P{pruned_ratio:.1f}_"
            f"T{mean_time:.2f}s.png"
        )
    else:
        save_name = (
            f"U{ulysses}_C{int(args.compile)}_{cache_type}_"
            f"R{args.rdt}_S{cached_stepes}_"
            f"T{mean_time:.2f}s.png"
        )
    image.save(save_name)
    logger.info(f"Image saved as {save_name}")

    if args.ulysses is not None:
        import torch.distributed as dist

        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")


if __name__ == "__main__":
    main()
