import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from utils import (
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
    MemoryTracker,
    print_rank0,
)
import cache_dit


args = get_args()
rank, device = maybe_init_distributed(args)
print_rank0(args)

model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get(
        "PIXART_SIGMA_DIR",
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    )
)
transformer = Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
pipe.to("cuda")

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)


pipe.set_progress_bar_config(disable=rank != 0)

# Set default prompt
prompt = "A small cactus with a happy face in the Sahara desert."
if args.prompt is not None:
    prompt = args.prompt


def run_pipe(warmup: bool = False):
    image = pipe(
        prompt,
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=50 if not warmup else 5,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]
    return image


# warmup
_ = run_pipe(warmup=True)


memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report(rank)

if rank == 0:
    stats = cache_dit.summary(pipe)
    time_cost = end - start
    save_path = f"pixart-sigma.{strify(args, stats)}.png"

    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
