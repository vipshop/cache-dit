import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel
from utils import (
    MemoryTracker,
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
)

import cache_dit

# NOTE: Only support context parallelism with 'native/_sdpa_cudnn' attn backend
# for Z-Image due to the attention mask in Z-Image is not None. Please use:
# `--parallel ulysses --attn native` or `--attn _sdpa_cudnn`.

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

pipe: ZImagePipeline = ZImagePipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "ZIMAGE_DIR",
            "Tongyi-MAI/Z-Image-Turbo",
        )
    ),
    torch_dtype=torch.bfloat16,
)

if args.quantize:
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
        quant_type=args.quantize_type,
        exclude_layers=[
            "noise_refiner",
            "context_refiner",
            "all_x_embedder",
            "all_final_layer",
            "t_embedder",
            "cap_embedder",
            "rope_embedder",
        ],
    )

if args.cache or args.parallel_type is not None:
    if args.cache:
        # Only warmup 4 steps (total 9 steps) for distilled models
        args.max_warmup_steps = min(4, args.max_warmup_steps)

    cachify(
        args,
        pipe,
        # Total 9 steps for distilled Z-Image-Turbo
        # e.g, 111110101, 1: compute, 0: dynamic cache
        steps_computation_mask=(
            cache_dit.steps_mask(
                compute_bins=[5, 1, 1],  # 7 steps compute
                cache_bins=[1, 1],  # max 2 steps cache
            )
            if args.steps_mask
            else None
        ),
    )

pipe.to(device)

assert isinstance(pipe.transformer, ZImageTransformer2DModel)

# Allow customize attention backend for Single GPU inference
if args.parallel_type is None:
    # native, flash, _native_cudnn, sage, etc.
    # _native_cudnn is faster than native(sdpa) on NVIDIA L20 with CUDA 12.9+.
    # '_sdpa_cudnn' is only in cache-dit to support context parallelism
    # with attn masks, e.g., ZImage. It is not in diffusers yet.
    if args.attn is not None:
        pipe.transformer.set_attention_backend(args.attn)

pipe.set_progress_bar_config(disable=rank != 0)

# Set default prompt
prompt = (
    "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
    "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
    "red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
    "lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. "
    "Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), "
    "blurred colorful distant lights."
)
if args.prompt is not None:
    prompt = args.prompt


def run_pipe(warmup: bool = False):
    image = pipe(
        prompt=prompt,
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=2 if warmup else (9 if args.steps is None else args.steps),
        guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


if args.compile:
    cache_dit.set_compile_configs()
    if args.compile_repeated_blocks:
        pipe.transformer.compile_repeated_blocks(
            mode="max-autotune-no-cudagraphs" if args.max_autotune else "default"
        )
    else:
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune-no-cudagraphs" if args.max_autotune else "default"
        )

# warmup
_ = run_pipe(warmup=True)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"zimage.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
