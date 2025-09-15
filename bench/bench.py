import os
import argparse
import torch
import random
import time
from PIL import Image
from tqdm import tqdm
from diffusers import FluxPipeline, FluxTransformer2DModel

import cache_dit

logger = cache_dit.init_logger(__name__)


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--cache-config", type=str, default=None)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=1)
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
    parser.add_argument("--save-dir", type=str, default="./tmp/drawbench")
    parser.add_argument(
        "--prompt-file", type=str, default="./prompts/DrawBench200.txt"
    )
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--test-num", type=int, default=None)
    return parser.parse_args()


def set_rand_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def init_pipeline(args) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Apply cache to the pipeline
    if args.cache or args.cache_config:
        if not args.use_block_adapter:
            if args.cache_config is None:
                cache_dit.enable_cache(
                    pipe,
                    # Cache context kwargs
                    Fn_compute_blocks=args.Fn_compute_blocks,
                    Bn_compute_blocks=args.Bn_compute_blocks,
                    max_warmup_steps=args.max_warmup_steps,
                    max_cached_steps=args.max_cached_steps,
                    max_continuous_cached_steps=args.max_continuous_cached_steps,
                    residual_diff_threshold=args.rdt,
                    l1_hidden_states_diff_threshold=(
                        None if not args.l1_diff else args.rdt
                    ),
                    enable_taylorseer=args.taylorseer,
                    enable_encoder_taylorseer=args.taylorseer,
                    taylorseer_cache_type="residual",
                    taylorseer_order=args.taylorseer_order,
                )
            else:
                cache_dit.enable_cache(
                    pipe, **cache_dit.load_options(args.cache_config)
                )
        else:
            assert isinstance(pipe.transformer, FluxTransformer2DModel)
            from cache_dit import ForwardPattern, BlockAdapter
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
                    Fn_compute_blocks=args.Fn_compute_blocks,
                    Bn_compute_blocks=args.Bn_compute_blocks,
                    max_warmup_steps=args.max_warmup_steps,
                    max_cached_steps=args.max_cached_steps,
                    max_continuous_cached_steps=args.max_continuous_cached_steps,
                    residual_diff_threshold=args.rdt,
                    l1_hidden_states_diff_threshold=(
                        None if not args.l1_diff else args.rdt
                    ),
                    enable_taylorseer=args.taylorseer,
                    enable_encoder_taylorseer=args.taylorseer,
                    taylorseer_cache_type="residual",
                    taylorseer_order=args.taylorseer_order,
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

    return pipe


def gen_image(args, pipe: FluxPipeline, prompt) -> Image.Image:
    assert prompt is not None
    image = pipe(
        prompt,
        num_inference_steps=args.steps,
        generator=torch.Generator(args.gen_device).manual_seed(args.seed),
    ).images[0]
    return image


@torch.no_grad()
def main():
    args = get_args()
    logger.info(f"Arguments: {args}")
    set_rand_seeds(args.seed)

    pipe = init_pipeline(args)
    pipe.set_progress_bar_config(disable=True)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    print(f"Loaded {len(prompts)} prompts from {args.prompt_file}")

    all_times = []
    base_dir = (
        f"C{int(args.compile)}_Q{int(args.quantize)}_{cache_dit.strify(pipe)}"
    )
    save_dir = os.path.join(args.save_dir, base_dir)
    os.makedirs(save_dir, exist_ok=True)

    if args.test_num is not None:
        prompts = prompts[: args.test_num]

    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        start = time.time()
        image = gen_image(args, pipe, prompt=prompt)
        end = time.time()
        all_times.append(end - start)
        save_name = os.path.join(save_dir, f"img_{i}.png")
        image.save(save_name)

    all_times.pop(0)  # Remove the first run time, usually warmup
    mean_time = sum(all_times) / len(all_times)
    logger.info(f"Mean Time: {mean_time:.2f}s")


if __name__ == "__main__":
    main()
