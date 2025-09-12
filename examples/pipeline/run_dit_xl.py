import os
import sys

sys.path.append("..")

import time
import torch

from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from utils import get_args, strify
import cache_dit

args = get_args()
print(args)


pipe = DiTPipeline.from_pretrained(
    os.environ.get(
        "DIT_XL_DIR",
        "facebook/DiT-XL-2-256",
    ),
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
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

words = ["white shark"]

class_ids = pipe.get_label_ids(words)

start = time.time()
image = pipe(
    class_labels=class_ids,
    num_inference_steps=25,
    generator=torch.Generator("cpu").manual_seed(33),
).images[0]
end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"dit-xl.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
