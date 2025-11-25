import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import AmusedPipeline
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)


pipe = AmusedPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "AMUSED_DIR",
            "amused/amused-512",
        )
    ),
    variant="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

prompt = "a photo of an astronaut riding a horse on mars"


if args.prompt is not None:

    prompt = args.prompt
memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(
    prompt,
    num_inference_steps=12,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"amused.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
