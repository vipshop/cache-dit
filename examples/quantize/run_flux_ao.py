import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from utils import get_args, strify, cachify
import cache_dit


args = get_args()
print(args)


pipe: FluxPipeline = FluxPipeline.from_pretrained(
    os.environ.get(
        "FLUX_DIR",
        "black-forest-labs/FLUX.1-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")


if args.cache:
    cachify(args, pipe)


if args.quantize:
    assert isinstance(pipe.transformer, FluxTransformer2DModel)
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
        quant_type=args.quantize_type,
    )


def run_pipe(pipe: FluxPipeline):
    image = pipe(
        "A cat holding a sign that says hello world",
        num_inference_steps=28,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


if args.compile:
    assert isinstance(pipe.transformer, FluxTransformer2DModel)
    pipe.transformer.compile_repeated_blocks()

    # warmup
    _ = run_pipe(pipe)


start = time.time()
image = run_pipe(pipe)
end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.ao.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
