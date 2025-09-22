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
    from cache_dit import BasicCacheConfig, TaylorSeerCalibratorConfig

    cache_dit.enable_cache(
        pipe,
        cache_config=BasicCacheConfig(
            Fn_compute_blocks=args.Fn,
            Bn_compute_blocks=args.Bn,
            max_warmup_steps=args.max_warmup_steps,
            max_cached_steps=args.max_cached_steps,
            max_continuous_cached_steps=args.max_continuous_cached_steps,
            residual_diff_threshold=args.rdt,
            enable_separate_cfg=False,
        ),
        calibrator_config=TaylorSeerCalibratorConfig(
            taylorseer_order=1,
        ),
    )
    print(cache_dit.strify(pipe))


start = time.time()
image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.{cache_dit.strify(pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
