import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)


pipe = FluxFillPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "FLUX_FILL_DIR",
            "black-forest-labs/FLUX.1-Fill-dev",
        )
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    cachify(args, pipe)

# Set default prompt
prompt = "a white paper cup"
if args.prompt is not None:
    prompt = args.prompt

if args.compile:
    from diffusers import FluxTransformer2DModel

    cache_dit.set_compile_configs()
    assert isinstance(pipe.transformer, FluxTransformer2DModel)
    pipe.transformer.compile_repeated_blocks(fullgraph=True)

    # warmup
    image = pipe(
        prompt=prompt,
        image=load_image("../data/cup.png"),
        mask_image=load_image("../data/cup_mask.png"),
        guidance_scale=30,
        num_inference_steps=28,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(
    prompt=prompt,
    image=load_image("../data/cup.png"),
    mask_image=load_image("../data/cup_mask.png"),
    guidance_scale=30,
    num_inference_steps=28,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux-fill.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
