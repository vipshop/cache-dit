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

if args.cache or args.parallel_type is not None:
    if args.cache:
        # Only warmup 4 steps (total 9 steps) for distilled models
        args.max_warmup_steps = min(4, args.max_warmup_steps)

    cachify(args, pipe)

pipe.to(device)

assert isinstance(pipe.transformer, ZImageTransformer2DModel)

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
    pipe.transformer = torch.compile(pipe.transformer)

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
