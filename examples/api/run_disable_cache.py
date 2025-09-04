import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline
from utils import get_args
import cache_dit


args = get_args()
print(args)


pipe = FluxPipeline.from_pretrained(
    os.environ.get(
        "FLUX_DIR",
        "black-forest-labs/FLUX.1-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")


if args.cache:
    adapter = cache_dit.enable_cache(pipe)
    # Test disable_cache api
    # cache_dit.disable_cache(adapter)
    cache_dit.disable_cache(pipe)


start = time.time()
image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

if args.cache:
    cache_dit.summary(adapter)

time_cost = end - start
save_path = f"flux.{cache_dit.strify(pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
