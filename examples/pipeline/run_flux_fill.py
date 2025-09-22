import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from utils import get_args, strify, cachify
import cache_dit


args = get_args()
print(args)


pipe = FluxFillPipeline.from_pretrained(
    os.environ.get(
        "FLUX_FILL_DIR",
        "black-forest-labs/FLUX.1-Fill-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    cachify(args, pipe)

start = time.time()
image = pipe(
    prompt="a white paper cup",
    image=load_image("../data/cup.png"),
    mask_image=load_image("../data/cup_mask.png"),
    guidance_scale=30,
    num_inference_steps=28,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux-fill.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
