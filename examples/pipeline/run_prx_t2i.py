import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import PRXPipeline
from utils import get_args, strify, cachify
import cache_dit

args = get_args()
print(args)

# Load pipeline with from_pretrained
pipe = PRXPipeline.from_pretrained(
    os.environ.get(
        "PRX_T2I_DIR",
        "Photoroom/prx-512-t2i-sft",
    ),
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

if args.cache:
    cachify(args, pipe)


def run_pipe():
    image = pipe(
        "A digital painting of a rusty, vintage tram on a sandy beach",
        num_inference_steps=28,
        guidance_scale=5.0,
    ).images[0]
    return image


start = time.time()
image = run_pipe()
end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"prx.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
