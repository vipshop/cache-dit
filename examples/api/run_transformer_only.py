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


pipe = FluxPipeline.from_pretrained(
    os.environ.get(
        "FLUX_DIR",
        "black-forest-labs/FLUX.1-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    from cache_dit import BlockAdapter, ForwardPattern

    assert isinstance(pipe.transformer, FluxTransformer2DModel)

    cachify(
        args,
        BlockAdapter(
            transformer=pipe.transformer,
            blocks=[
                pipe.transformer.transformer_blocks,
                pipe.transformer.single_transformer_blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_1,
                ForwardPattern.Pattern_1,
            ],
            check_forward_pattern=True,
        ),
    )


def run_pipe():
    image = pipe(
        "A cat holding a sign that says hello world",
        num_inference_steps=28,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


# warmup
_ = run_pipe()

start = time.time()
image = run_pipe()
end = time.time()

cache_dit.summary(pipe.transformer)

time_cost = end - start
save_path = f"flux.{strify(args, pipe.transformer)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
