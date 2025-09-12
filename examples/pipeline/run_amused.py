import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import AmusedPipeline
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)


pipe = AmusedPipeline.from_pretrained(
    os.environ.get(
        "AMUSED_DIR",
        "amused/amused-512",
    ),
    variant="fp16",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

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


prompt = "a photo of an astronaut riding a horse on mars"

start = time.time()
image = pipe(
    prompt,
    num_inference_steps=12,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"amused.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
