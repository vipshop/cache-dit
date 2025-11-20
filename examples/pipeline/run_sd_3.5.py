import os
import sys

sys.path.append("..")

import time
import torch

from diffusers import StableDiffusion3Pipeline
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit

args = get_args()
print(args)

model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get(
        "SD_3_5_DIR",
        "stabilityai/stable-diffusion-3.5-large",
    )
)

pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="balanced",
)

if args.cache:
    cachify(args, pipe)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
if args.prompt is not None:
    prompt = args.prompt
image = pipe(
    prompt,
    num_inference_steps=50,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)
time_cost = end - start
save_path = f"sd_3_5.{strify(args, stats)}.png"

print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
