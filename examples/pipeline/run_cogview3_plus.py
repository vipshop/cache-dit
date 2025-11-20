import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import CogView3PlusPipeline
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit

args = get_args()
print(args)


pipe = CogView3PlusPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "COGVIEW3_DIR",
            "THUDM/CogView3-Plus-3B",
        )
    ),
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."
if args.prompt is not None:
    prompt = args.prompt

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(
    prompt=prompt,
    guidance_scale=7.0,
    num_inference_steps=50,
    width=1024,
    height=1024,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"cogview3_plus.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
