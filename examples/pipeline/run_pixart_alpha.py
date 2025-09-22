import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import PixArtAlphaPipeline
from utils import get_args, strify, cachify
import cache_dit


args = get_args()
print(args)

model_id = os.environ.get(
    "PIXART_ALPHA_DIR",
    "PixArt-alpha/PixArt-XL-2-1024-MS",
)

pipe = PixArtAlphaPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

start = time.time()
prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(
    prompt,
    num_inference_steps=50,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)
time_cost = end - start
save_path = f"pixart-alpha.{strify(args, stats)}.png"

print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
