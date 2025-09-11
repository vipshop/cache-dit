import os
import sys

sys.path.append("..")

import time
import torch

from diffusers import AuraFlowPipeline
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)

model_id = os.environ.get("AURAFLOW_DIR", "fal/AuraFlow-v0.3")

pipe = AuraFlowPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
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
    prompt="rempage of the iguana character riding F1, fast and furious, cinematic movie poster",
    width=1536,
    height=768,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(1),
    guidance_scale=3.5,
).images[0]

end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"auraflow.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
