import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import PRXPipeline
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit

args = get_args()
print(args)

# Load pipeline with from_pretrained
pipe = PRXPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "PRX_T2I_DIR",
            "Photoroom/prx-512-t2i-sft",
        )
    ),
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

# Set default prompt
prompt = "A digital painting of a rusty, vintage tram on a sandy beach"
if args.prompt is not None:
    prompt = args.prompt


def run_pipe():
    image = pipe(
        prompt,
        num_inference_steps=28,
        guidance_scale=5.0,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"prx.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
