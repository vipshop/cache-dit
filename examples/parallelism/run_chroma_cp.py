import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import ChromaPipeline
from utils import (
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
    MemoryTracker,
    print_rank0,
)
import cache_dit

# NOTE: Please use `--parallel ulysses --attn naitve` for Chroma with context parallelism,

args = get_args()
rank, device = maybe_init_distributed(args)
print_rank0(args)

pipe = ChromaPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get("CHROMA1_DIR", "lodestones/Chroma1-HD")
    ),
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

prompt = [
    "A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done."
]
if args.prompt is not None:
    prompt = [args.prompt]

negative_prompt = [
    "low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"
]
if args.negative_prompt is not None:
    negative_prompt = [args.negative_prompt]


def run_pipe(warmup: bool = False):
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=torch.Generator("cpu").manual_seed(433),
        num_inference_steps=40 if not warmup else 5,
        guidance_scale=3.0,
        num_images_per_prompt=1,
    ).images[0]
    return image


# warmup
run_pipe(warmup=True)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report(rank)

if rank == 0:
    cache_dit.summary(pipe, details=True)

    time_cost = end - start
    save_path = f"chroma1-hd.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
