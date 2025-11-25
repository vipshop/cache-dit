import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)


pipe = FluxKontextPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "FLUX_KONTEXT_DIR",
            "black-forest-labs/FLUX.1-Kontext-dev",
        )
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    cachify(args, pipe)

# Set default prompt
prompt = "Add a hat to the cat"
if args.prompt is not None:
    prompt = args.prompt

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()

image = pipe(
    image=load_image("../data/cat.png"),
    prompt=prompt,
    guidance_scale=2.5,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux-kontext.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
