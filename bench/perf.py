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
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--cache-config", type=str, default=None)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=2)
    parser.add_argument("--l1-diff", action="store_true", default=False)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=1)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--max-warmup-steps", type=int, default=0)
    parser.add_argument("--max-cached-steps", type=int, default=-1)
    parser.add_argument("--max-continuous-cached-steps", type=int, default=-1)
    parser.add_argument("--gen-device", type=str, default="cpu")
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--inductor-flags", action="store_true", default=False)
    parser.add_argument("--compile-all", action="store_true", default=False)
    parser.add_argument("--quantize", "-q", action="store_true", default=False)
    parser.add_argument(
        "--use-block-adapter", "--adapt", action="store_true", default=False
    )
    parser.add_argument(
        "--use-auto-block-adapter", "--auto", action="store_true", default=False
    )
    parser.add_argument("--save-dir", type=str, default="./tmp")
    return parser.parse_args()


def set_rand_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def main():
    args = get_args()
    logger.info(f"Arguments: {args}")
    set_rand_seeds(args.seed)

    pipe = FluxPipeline.from_pretrained(
        os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Apply cache to the pipeline
    if args.cache or args.cache_config:
        if not args.use_block_adapter:
            if args.cache_config is None:
                from cache_dit import (
                    BasicCacheConfig,
                    TaylorSeerCalibratorConfig,
                )

                cache_dit.enable_cache(
                    pipe,
                    # Cache context kwargs
                    cache_config=BasicCacheConfig(
                        Fn_compute_blocks=args.Fn_compute_blocks,
                        Bn_compute_blocks=args.Bn_compute_blocks,
                        max_warmup_steps=args.max_warmup_steps,
                        max_cached_steps=args.max_cached_steps,
                        max_continuous_cached_steps=args.max_continuous_cached_steps,
                        residual_diff_threshold=args.rdt,
                    ),
                    calibrator_config=(
                        TaylorSeerCalibratorConfig(
                            taylorseer_order=args.taylorseer_order,
                        )
                        if args.taylorseer
                        else None
                    ),
                )
            else:
                cache_dit.enable_cache(
                    pipe, **cache_dit.load_options(args.cache_config)
                )
        else:
            assert isinstance(pipe.transformer, FluxTransformer2DModel)
            from cache_dit import (
                ForwardPattern,
                BlockAdapter,
                BasicCacheConfig,
                TaylorSeerCalibratorConfig,
            )
            from cache_dit.cache_factory.patch_functors import FluxPatchFunctor

            if args.cache_config is None:

                cache_dit.enable_cache(
                    # BlockAdapter & forward pattern
                    (
                        BlockAdapter(
                            pipe=pipe,
                            transformer=pipe.transformer,
                            blocks=(
                                pipe.transformer.transformer_blocks
                                + pipe.transformer.single_transformer_blocks
                            ),
                            blocks_name="transformer_blocks",
                            dummy_blocks_names=["single_transformer_blocks"],
                            patch_functor=FluxPatchFunctor(),
                            forward_pattern=ForwardPattern.Pattern_1,
                        )
                        if not args.use_auto_block_adapter
                        else BlockAdapter(
                            pipe=pipe,
                            auto=True,
                            blocks_policy="min",
                            patch_functor=FluxPatchFunctor(),
                            forward_pattern=ForwardPattern.Pattern_1,
                        )
                    ),
                    # Cache context kwargs
                    cache_config=BasicCacheConfig(
                        Fn_compute_blocks=args.Fn_compute_blocks,
                        Bn_compute_blocks=args.Bn_compute_blocks,
                        max_warmup_steps=args.max_warmup_steps,
                        max_cached_steps=args.max_cached_steps,
                        max_continuous_cached_steps=args.max_continuous_cached_steps,
                        residual_diff_threshold=args.rdt,
                    ),
                    calibrator_config=(
                        TaylorSeerCalibratorConfig(
                            taylorseer_order=args.taylorseer_order,
                        )
                        if args.taylorseer
                        else None
                    ),
                )
            else:
                cache_dit.enable_cache(
                    # BlockAdapter & forward pattern
                    (
                        BlockAdapter(
                            pipe,
                            transformer=pipe.transformer,
                            blocks=(
                                pipe.transformer.transformer_blocks
                                + pipe.transformer.single_transformer_blocks
                            ),
                            blocks_name="transformer_blocks",
                            dummy_blocks_names=["single_transformer_blocks"],
                            patch_functor=FluxPatchFunctor(),
                            forward_pattern=ForwardPattern.Pattern_1,
                        )
                        if not args.use_auto_block_adapter
                        else BlockAdapter(
                            pipe=pipe,
                            auto=True,
                            blocks_policy="min",
                            patch_functor=FluxPatchFunctor(),
                            forward_pattern=ForwardPattern.Pattern_1,
                        )
                    ),
                    # Cache context kwargs
                    **cache_dit.load_options(args.cache_config),
                )

    if args.quantize:
        # Apply Quantization (default: FP8 DQ) to Transformer
        pipe.transformer = cache_dit.quantize(pipe.transformer)

    if args.compile or args.quantize:
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

    cache_dit.summary(pipe, details=True)
    save_name = (
        f"C{int(args.compile)}_Q{int(args.quantize)}_"
        f"{cache_dit.strify(pipe)}_"
        f"T{mean_time:.2f}s.png"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    save_name = os.path.join(args.save_dir, save_name)

    image.save(save_name)
    logger.info(f"Image saved as {save_name}")


if __name__ == "__main__":
    main()
