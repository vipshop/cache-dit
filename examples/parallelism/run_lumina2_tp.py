import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import Lumina2Pipeline, Lumina2Transformer2DModel
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

pipe: Lumina2Pipeline = Lumina2Pipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get("LUMINA_DIR", "Alpha-VLLM/Lumina-Image-2.0")
    ),
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

pipe.to(device)

assert isinstance(pipe.transformer, Lumina2Transformer2DModel)

pipe.set_progress_bar_config(disable=rank != 0)

# Set default prompt
prompt = (
    "A serene photograph capturing the golden reflection of the sun on a vast"
    " expanse of water. The sun is positioned at the top center, casting a brilliant, "
    "shimmering trail of light across the rippling surface. The water is textured "
    "with gentle waves, creating a rhythmic pattern that leads the eye towards "
    "the horizon. The entire scene is bathed in warm, golden hues, enhancing the "
    "tranquil and meditative atmosphere. High contrast, natural lighting, golden hour, "
    "photorealistic, expansive composition, reflective surface, peaceful, "
    "visually harmonious."
)

if args.prompt is not None:
    prompt = args.prompt


def run_pipe(warmup: bool = False):
    image = pipe(
        prompt=prompt,
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=5 if warmup else (50 if args.steps is None else args.steps),
        guidance_scale=4.0,
        cfg_trunc_ratio=0.25,
        cfg_normalization=True,
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
    save_path = f"lumina2.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
