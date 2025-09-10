import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import ChromaPipeline
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)


pipe = ChromaPipeline.from_pretrained(
    os.environ.get(
        "CHROMA1_DIR",
        "lodestones/Chroma1-HD",
    ),
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")

if args.cache:
    cache_dit.enable_cache(
        pipe,
        # Cache context kwargs
        Fn_compute_blocks=args.Fn,
        Bn_compute_blocks=args.Bn,
        max_warmup_steps=args.max_warmup_steps,
        max_cached_steps=args.max_cached_steps,
        max_continuous_cached_steps=args.max_continuous_cached_steps,
        enable_taylorseer=args.taylorseer,
        enable_encoder_taylorseer=args.taylorseer,
        taylorseer_order=args.taylorseer_order,
        residual_diff_threshold=args.rdt,
    )

prompt = [
    "A high-fashion close-up portrait of a blonde woman in clear sunglasses. The image uses a bold teal and red color split for dramatic lighting. The background is a simple teal-green. The photo is sharp and well-composed, and is designed for viewing with anaglyph 3D glasses for optimal effect. It looks professionally done."
]
negative_prompt = [
    "low quality, ugly, unfinished, out of focus, deformed, disfigure, blurry, smudged, restricted palette, flat colors"
]

start = time.time()
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.Generator("cpu").manual_seed(433),
    num_inference_steps=40,
    guidance_scale=3.0,
    num_images_per_prompt=1,
).images[0]
end = time.time()

cache_dit.summary(pipe, details=True)

time_cost = end - start
save_path = f"chroma1-hd.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
