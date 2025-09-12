import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import SanaPipeline
from utils import get_args, strify
import cache_dit

args = get_args()
print(args)

model_id = os.environ.get(
    "SANA_DIR", "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers"
)

pipe = SanaPipeline.from_pretrained(
    model_id,
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


prompt = "a tiny astronaut hatching from an egg on the moon"

start = time.time()
image = pipe(
    prompt,
    num_inference_steps=20,
    generator=torch.Generator("cpu").manual_seed(1),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"sana.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
