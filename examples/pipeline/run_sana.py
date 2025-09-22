import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import SanaPipeline
from utils import get_args, strify, cachify
import cache_dit

args = get_args()
print(args)

model_id = os.environ.get(
    "SANA_DIR", "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
)

pipe = SanaPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    cachify(args, pipe)

prompt = "a tiny astronaut hatching from an egg on the moon"

start = time.time()
image = pipe(
    prompt,
    num_inference_steps=20,
    generator=torch.Generator("cpu").manual_seed(1),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"sana.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
