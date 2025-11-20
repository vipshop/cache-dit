import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline
from utils import get_args, MemoryTracker
import cache_dit


args = get_args()
print(args)


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
).to("cuda")


if args.cache:
    adapter = cache_dit.enable_cache(pipe)
    print(cache_dit.strify(pipe))
    # Test disable_cache api
    # cache_dit.disable_cache(adapter)
    cache_dit.disable_cache(pipe)
    print(cache_dit.strify(pipe))


# Set default prompt
prompt = "A cat holding a sign that says hello world"
if args.prompt is not None:
    prompt = args.prompt

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(
    prompt,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.{cache_dit.strify(pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
