import os
import sys

sys.path.append("..")

import time
import torch

from diffusers import AuraFlowPipeline
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)

model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get("AURAFLOW_DIR", "fal/AuraFlow-v0.3")
)

pipe = AuraFlowPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

if args.cache:
    cachify(args, pipe)

# Set default prompt
prompt = "rempage of the iguana character riding F1, fast and furious, cinematic movie poster"
if args.prompt is not None:
    prompt = args.prompt

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(
    prompt=prompt,
    width=1536,
    height=768,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(1),
    guidance_scale=3.5,
).images[0]

end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"auraflow.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
