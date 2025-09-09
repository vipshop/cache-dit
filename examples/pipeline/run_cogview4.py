import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import CogView4Pipeline
from utils import get_args, strify
import cache_dit

args = get_args()
print(args)


pipe = CogView4Pipeline.from_pretrained(
    os.environ.get(
        "COGVIEW4_DIR",
        "THUDM/CogView4-6B",
    ),
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")

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
        enable_spearate_cfg=True,
    )


prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."

start = time.time()
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,  # >1, do separate cfg
    num_inference_steps=50,
    width=1024,
    height=1024,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"cogview4.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
