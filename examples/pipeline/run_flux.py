import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline
from utils import get_args, strify, cachify
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
    cachify(args, pipe)


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
save_path = f"flux.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
