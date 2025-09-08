import os
import sys

sys.path.append("..")

import time
import torch

from diffusers import StableDiffusion3Pipeline
from utils import get_args, strify
import cache_dit

args = get_args()
print(args)

model_id = os.environ.get(
    "SD_3_5_DIR",
    "stabilityai/stable-diffusion-3.5-large",
)

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)

if args.cache:
    cache_dit.enable_cache(pipe)

start = time.time()
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(
    prompt,
    num_inference_steps=50,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

end = time.time()

stats = cache_dit.summary(pipe)
time_cost = end - start
save_path = f"sd_3_5.{strify(args, stats)}.png"

print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
