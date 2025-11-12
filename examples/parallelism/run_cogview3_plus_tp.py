import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import CogView3PlusPipeline
from utils import (
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
)
import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

pipe = CogView3PlusPipeline.from_pretrained(
    os.environ.get(
        "COGVIEW3_DIR",
        "THUDM/CogView3-Plus-3B",
    ),
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)
torch.cuda.empty_cache()
pipe.enable_model_cpu_offload(device=device)

prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."


def run_pipe(warmup: bool = False):
    image = pipe(
        prompt=prompt,
        guidance_scale=7.0,
        num_inference_steps=50 if not warmup else 5,
        width=1024,
        height=1024,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


# warmup
run_pipe(warmup=True)

start = time.time()
image = run_pipe()
end = time.time()

if rank == 0:
    stats = cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"cogview3_plus_tp.{strify(args, stats)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
