import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline, PixArtPipeline
from utils import (
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
)
import cache_dit


args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

# Support both PixArt-Alpha and PixArt-Sigma models
model_id = os.environ.get(
    "PIXART_DIR",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    # Alternative models:
    # "PixArt-alpha/PixArt-XL-2-1024-MS",
    # "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
)

# Determine pipeline type based on model
if "Sigma" in model_id:
    pipeline_class = PixArtSigmaPipeline
else:
    pipeline_class = PixArtPipeline

transformer = Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)

pipe = pipeline_class.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(warmup: bool = False):
    image = pipe(
        "A small cactus with a happy face in the Sahara desert.",
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=50 if not warmup else 5,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]
    return image


# Warmup
_ = run_pipe(warmup=True)

start = time.time()
image = run_pipe()
end = time.time()

if rank == 0:
    stats = cache_dit.summary(pipe)
    time_cost = end - start
    model_name = "pixart-sigma" if "Sigma" in model_id else "pixart-alpha"
    save_path = f"{model_name}.{strify(args, stats)}.png"

    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
