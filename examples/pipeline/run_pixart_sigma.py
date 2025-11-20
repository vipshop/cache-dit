import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)

model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get(
        "PIXART_SIGMA_DIR",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    )
)
transformer = Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
prompt = "A small cactus with a happy face in the Sahara desert."
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
save_path = f"pixart-sigma.{strify(args, stats)}.png"

print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
