import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Lumina2Transformer2DModel, Lumina2Pipeline
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)


model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get("LUMINA_DIR", "Alpha-VLLM/Lumina-Image-2.0")
)

ckpt_path = os.path.join(model_id, "consolidated.00-of-01.pth")
transformer = Lumina2Transformer2DModel.from_single_file(ckpt_path, torch_dtype=torch.bfloat16)

pipe = Lumina2Pipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch.bfloat16
)

pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

# Set default prompt
prompt = "a cute cat holding a sign that says hello 'Lumina2'"
if args.prompt is not None:
    prompt = args.prompt

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(
    prompt,
    height=1024,
    width=1024,
    num_inference_steps=30,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"lumina2.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
