import os
import torch
from diffusers import FluxFillPipeline
from diffusers.utils import load_image
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

pipe = FluxFillPipeline.from_pretrained(
    os.environ.get(
        "FLUX_FILL_DIR",
        "black-forest-labs/FLUX.1-Fill-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")


# Default options, F8B8, good balance between performance and precision
cache_options = CacheType.default_options(CacheType.DBCache)

apply_cache_on_pipe(pipe, **cache_options)

image = pipe(
    prompt="a white paper cup",
    image=load_image("data/cup.png"),
    mask_image=load_image("data/cup_mask.png"),
    guidance_scale=30,
    num_inference_steps=28,
    max_sequence_length=512,
    generator=torch.Generator("cuda").manual_seed(0),
).images[0]

print("Saving image to flux-fill.png")
image.save("flux-fill.png")
