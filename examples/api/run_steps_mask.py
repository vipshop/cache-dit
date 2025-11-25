import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from utils import get_args, strify, MemoryTracker
import cache_dit


parser = get_args(parse=False)
parser.add_argument(
    "--step-mask",
    type=str,
    default="slow",
    choices=["slow", "medium", "fast", "ultra", "s", "m", "f", "u"],
)
parser.add_argument(
    "--step-policy",
    type=str,
    default="dynamic",
    choices=["dynamic", "static"],
)
args = parser.parse_args()
print(args)


step_mask_aliases = {
    "s": "slow",
    "m": "medium",
    "f": "fast",
    "u": "ultra",
}
if args.step_mask in step_mask_aliases:
    args.step_mask = step_mask_aliases[args.step_mask]

# Define different step computation masks for 28 steps
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
    "ultra": cache_dit.steps_mask(
        compute_bins=[4, 1, 1, 1, 1],  # 8
        cache_bins=[1, 4, 5, 6, 6],  # 20
    ),
}

step_computation_dynamic_policy_rdt = {
    "slow": 0.20,
    "medium": 0.25,
    "fast": 0.30,
    "ultra": 0.34,
}

if args.rdt == 0.08:  # default
    args.rdt = step_computation_dynamic_policy_rdt[args.step_mask]


pipe = FluxPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "FLUX_DIR",
            "black-forest-labs/FLUX.1-dev",
        )
    ),
    torch_dtype=torch.bfloat16,
)

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

assert isinstance(pipe.transformer, FluxTransformer2DModel)
if args.quantize:
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
        quant_type=args.quantize_type,
        exclude_layers=[
            "embedder",
            "embed",
        ],
    )
    pipe.text_encoder_2 = cache_dit.quantize(
        pipe.text_encoder_2,
        quant_type=args.quantize_type,
    )
    print(f"Applied quantization: {args.quantize_type} to Transformer and Text Encoder 2.")

pipe.to("cuda")

if args.attn is not None:
    if hasattr(pipe.transformer, "set_attention_backend"):
        pipe.transformer.set_attention_backend(args.attn)
        print(f"Set attention backend to {args.attn}")


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)
    pipe.text_encoder = torch.compile(pipe.text_encoder)
    pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2)
    pipe.vae = torch.compile(pipe.vae)

# Set default prompt
prompt = "A cat holding a sign that says hello world"
if args.prompt is not None:
    prompt = args.prompt


def run_pipe():
    image = pipe(
        prompt,
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=28 if args.steps is None else args.steps,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


# warmup
_ = run_pipe()

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)

# python3 run_steps_mask.py --cache --Fn 1 --step-mask s --step-policy static
# python3 run_steps_mask.py --cache --Fn 1 --step-mask s --step-policy dynamic
# python3 run_steps_mask.py --cache --Fn 1 --step-mask m --step-policy dynamic
# python3 run_steps_mask.py --cache --Fn 1 --step-mask f --step-policy dynamic
# python3 run_steps_mask.py --cache --Fn 1 --step-mask f --step-policy dynamic --taylorseer --taylorseer-order 1
# python3 run_steps_mask.py --cache --Fn 1 --step-mask u --step-policy dynamic
# python3 run_steps_mask.py --cache --Fn 1 --step-mask u --step-policy dynamic --taylorseer --taylorseer-order 1
# python3 run_steps_mask.py --cache --Fn 1 --step-mask u --step-policy dynamic --compile --taylorseer --taylorseer-order 1
# python3 run_steps_mask.py --cache --Fn 1 --step-mask u --step-policy dynamic --compile --taylorseer --taylorseer-order 1 --quantize --quantize-type float8 --attn sage
