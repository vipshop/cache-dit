import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline
from utils import get_args
import cache_dit


parser = get_args(parse=False)
parser.add_argument("--step-mask", type=str, default="slow", choices=["slow", "medium", "fast"])
parser.add_argument("--step-policy", type=str, default="static", choices=["dynamic", "static"])
args = parser.parse_args()
print(args)


step_computation_masks = {
    "slow": cache_dit.steps_mask(
        compute_bins=[8, 3, 3, 2, 2],  # 18
        cache_bins=[1, 2, 2, 2, 3],  # 10
    ),
    "medium": cache_dit.steps_mask(
        compute_bins=[6, 2, 2, 2, 2],  # 14
        cache_bins=[1, 3, 3, 3, 4],  # 14
    ),
    "fast": cache_dit.steps_mask(
        compute_bins=[6, 1, 1, 1, 1],  # 10
        cache_bins=[1, 3, 4, 5, 5],  # 18
    ),
}

pipe = FluxPipeline.from_pretrained(
    os.environ.get(
        "FLUX_DIR",
        "black-forest-labs/FLUX.1-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig

    # Scheme: Hybrid DBCache + LeMiCa/EasyCache + TaylorSeer
    cache_dit.enable_cache(
        pipe,
        cache_config=DBCacheConfig(
            # Basic DBCache configs
            Fn_compute_blocks=args.Fn,
            Bn_compute_blocks=args.Bn,
            max_warmup_steps=args.max_warmup_steps,
            warmup_interval=args.warmup_interval,
            max_cached_steps=args.max_cached_steps,
            max_continuous_cached_steps=args.max_continuous_cached_steps,
            residual_diff_threshold=args.rdt,
            # LeMiCa or EasyCache style Mask for 28 steps, e.g,
            # 111111010010000010000100001, 1: compute, 0: cache.
            steps_computation_mask=step_computation_masks[args.step_mask],
            # The policy for cache steps can be 'dynamic' or 'static'
            steps_computation_policy=args.step_policy,
        ),
        calibrator_config=(
            TaylorSeerCalibratorConfig(
                taylorseer_order=args.taylorseer_order,
            )
            if args.taylorseer
            else None
        ),
    )


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)


def run_pipe():
    image = pipe(
        "A cat holding a sign that says hello world",
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=28 if args.steps is None else args.steps,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


# warmup
_ = run_pipe()

start = time.time()
image = run_pipe()
end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.{cache_dit.strify(pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
