import os
import sys

sys.path.append("..")

import time
import torch

from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit

args = get_args()
print(args)


pipe = DiTPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "DIT_XL_DIR",
            "facebook/DiT-XL-2-256",
        )
    ),
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

words = ["white shark"]

class_ids = pipe.get_label_ids(words)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(
    class_labels=class_ids,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(33),
).images[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"dit-xl.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
