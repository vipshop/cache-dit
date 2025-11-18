import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline
from utils import get_args
import cache_dit


args = get_args()
print(args)


pipe = FluxPipeline.from_pretrained(
    os.environ.get(
        "FLUX_DIR",
        "black-forest-labs/FLUX.1-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    from cache_dit import DBCacheConfig

    cache_dit.enable_cache(
        pipe,
        cache_config=DBCacheConfig(
            Fn_compute_blocks=1,
            Bn_compute_blocks=0,
            max_warmup_steps=6,
            warmup_interval=1,
            residual_diff_threshold=args.rdt,  # 0.08 default
            # Mask for 30 steps, 11111101001000001000001000001
            steps_computation_mask=cache_dit.steps_mask(
                compute_bins=[6, 1, 1, 1, 1],  # 10
                cache_bins=[1, 2, 5, 6, 6],  # 20
            ),
            # The policy for cache steps can be 'dynamic' or 'static'
            steps_computation_policy="dynamic",
        ),
    )
    print(cache_dit.strify(pipe))


start = time.time()
image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=30,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.{cache_dit.strify(pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
