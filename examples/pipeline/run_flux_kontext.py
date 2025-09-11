import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)


pipe = FluxKontextPipeline.from_pretrained(
    os.environ.get(
        "FLUX_KONTEXT_DIR",
        "black-forest-labs/FLUX.1-Kontext-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")


if args.cache:
    cache_dit.enable_cache(
        pipe,
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


start = time.time()

image = pipe(
    image=load_image("../data/cat.png"),
    prompt="Add a hat to the cat",
    guidance_scale=2.5,
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux-kontext.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
