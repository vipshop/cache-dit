import os
import argparse
import functools
import torch
import random
import time
from PIL import Image
from tqdm import tqdm
from diffusers import FluxPipeline, FluxTransformer2DModel

try:
    from utils import calculate_flops

    CALFLOPS_AVAILABLE = True
except ImportError:
    calculate_flops = None
    CALFLOPS_AVAILABLE = False

import cache_dit

logger = cache_dit.init_logger(__name__)


def set_rand_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)


# TODO: Support calculate_flops using functools
total_flops = 0
total_steps = 0
all_tflops = []


def apply_flops_hook(
    args: argparse.Namespace,
    transformer: FluxTransformer2DModel,
):

    old_forward = transformer.forward

    @functools.wraps(old_forward)
    def new_forward(self: FluxTransformer2DModel, **kwargs):
        global total_flops, total_steps, all_tflops

        # reset original forward to avoid maximum recursion depth error.
        hook_forward = transformer.forward
        transformer.forward = old_forward.__get__(transformer)
        step_flops, _, _, results = calculate_flops(
            model=transformer, kwargs=kwargs
        )
        transformer.forward = hook_forward.__get__(transformer)

        total_flops += step_flops
        total_steps += 1

        # [1, 50]: one inference
        if total_steps % args.steps == 0:
            if total_steps > 0:
                total_tflops = total_flops * 10 ** (-12)
                all_tflops.append(total_tflops)
                logger.debug(f"Total FLOPs: {total_tflops} TFLOPs")
            total_flops = 0  # reset

        return results

    transformer.forward = new_forward.__get__(transformer)
    logger.info("Applied FLOPs hook to transformer!")

    return transformer


def init_flux_pipe(args: argparse.Namespace) -> FluxPipeline:
    pipe: FluxPipeline = FluxPipeline.from_pretrained(
        os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Apply cache to the pipeline
    if args.cache:
        if args.disable_block_adapter:
            cache_dit.enable_cache(
                pipe,
                # Cache context kwargs
                Fn_compute_blocks=args.Fn_compute_blocks,
                Bn_compute_blocks=args.Bn_compute_blocks,
                max_warmup_steps=args.max_warmup_steps,
                max_cached_steps=args.max_cached_steps,
                max_continuous_cached_steps=args.max_continuous_cached_steps,
                residual_diff_threshold=args.rdt,
                enable_taylorseer=args.taylorseer,
                enable_encoder_taylorseer=args.taylorseer,
                taylorseer_cache_type="residual",
                taylorseer_order=args.taylorseer_order,
            )
        else:
            assert isinstance(pipe.transformer, FluxTransformer2DModel)
            from cache_dit import ForwardPattern, BlockAdapter
            from cache_dit.cache_factory.patch_functors import FluxPatchFunctor

            cache_dit.enable_cache(
                # BlockAdapter & forward pattern
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
                ),
                # Cache context kwargs
                Fn_compute_blocks=args.Fn_compute_blocks,
                Bn_compute_blocks=args.Bn_compute_blocks,
                max_warmup_steps=args.max_warmup_steps,
                max_cached_steps=args.max_cached_steps,
                max_continuous_cached_steps=args.max_continuous_cached_steps,
                residual_diff_threshold=args.rdt,
                enable_taylorseer=args.taylorseer,
                enable_encoder_taylorseer=args.taylorseer,
                taylorseer_cache_type="residual",
                taylorseer_order=args.taylorseer_order,
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
        if not args.compile_all:
            logger.warning(
                "Only compile transformer blocks not the whole model "
                "for FluxTransformer2DModel to keep higher precision."
            )
            pipe.transformer.compile_repeated_blocks(fullgraph=True)
        else:
            pipe.transformer = torch.compile(pipe.transformer, mode="default")

    if args.cal_flops:
        pipe.transformer = apply_flops_hook(args, pipe.transformer)

    return pipe


def gen_flux_image(
    args: argparse.Namespace, pipe: FluxPipeline, prompt: str = None
) -> Image.Image:
    assert prompt is not None
    image = pipe(
        prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    return image


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    # Cache params
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=1)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=8)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--max-warmup-steps", "--w", type=int, default=8)
    parser.add_argument("--max-cached-steps", "--mc", type=int, default=-1)
    parser.add_argument(
        "--max-continuous-cached-steps", "--mcc", type=int, default=-1
    )
    parser.add_argument(
        "--disable-block-adapter", action="store_true", default=False
    )
    # Compile & FP8
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--inductor-flags", action="store_true", default=False)
    parser.add_argument("--compile-all", action="store_true", default=False)
    parser.add_argument("--quantize", "--q", action="store_true", default=False)
    # Test data
    parser.add_argument("--save-dir", type=str, default="./tmp/DrawBench200")
    parser.add_argument(
        "--prompt-file", type=str, default="./prompts/DrawBench200.txt"
    )
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--test-num", type=int, default=None)
    parser.add_argument(
        "--cal-flops", "--flops", action="store_true", default=False
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    # TODO: Support more pipelines, such as Qwen-Image, DiT-XL, etc.
    args = get_args()
    logger.info(f"Arguments: {args}")
    set_rand_seeds(args.seed)

    pipe = init_flux_pipe(args)
    pipe.set_progress_bar_config(disable=True)

    # Load prompts
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f.readlines() if line.strip()]
    if args.test_num is not None:
        prompts = prompts[: args.test_num]
    logger.info(f"Loaded {len(prompts)} prompts from: {args.prompt_file}")

    all_times = []
    perf_tag = (
        f"C{int(args.compile)}_Q{int(args.quantize)}_{cache_dit.strify(pipe)}"
    )
    save_dir = os.path.join(args.save_dir, perf_tag)
    os.makedirs(save_dir, exist_ok=True)

    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        start = time.time()
        image = gen_flux_image(args, pipe, prompt=prompt)
        end = time.time()
        all_times.append(end - start)
        save_name = os.path.join(save_dir, f"img_{i}.png")
        image.save(save_name)

    # Perf. Latency and TFLOPs
    if len(all_times) > 1:
        all_times.pop(0)  # Remove the first run time, usually warmup
    mean_time = sum(all_times) / len(all_times)
    perf_msg = f"Perf. {perf_tag}, Mean pipeline time: {mean_time:.2f}s"
    if args.cal_flops and len(all_tflops) > 0:
        mean_tflops = sum(all_tflops) / len(all_tflops)
        perf_msg += f", Mean pipeline TFLOPs: {mean_tflops:.2f}"
    logger.info(perf_msg)


if __name__ == "__main__":
    main()
    # python3 bench.py --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --rdt 0.08
    # python3 bench.py --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --rdt 0.08
    # python3 bench.py --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --rdt 0.08
    # python3 bench.py --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --mcc 3 --rdt 0.16
    # python3 bench.py --cache --Fn 8 --Bn 0 --max-warmup-steps 8 --mcc 2 --rdt 0.16
    # python3 bench.py --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --mcc 3 --rdt 0.16
    # python3 bench.py --cache --Fn 8 --Bn 0 --max-warmup-steps 4 --mcc 2 --rdt 0.16
    # python3 bench.py --cache --Fn 4 --Bn 0 --max-warmup-steps 4 --mcc 2 --rdt 0.16
