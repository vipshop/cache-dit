import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from utils import get_args, strify
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
    from cache_dit import (
        BlockAdapter,
        ForwardPattern,
        ParamsModifier,
        DBCacheConfig,
    )

    assert isinstance(pipe.transformer, FluxTransformer2DModel)

    cache_dit.enable_cache(
        BlockAdapter(
            pipe=None,
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
        cache_config=(
            DBCacheConfig(
                Fn_compute_blocks=args.Fn,
                Bn_compute_blocks=args.Bn,
                max_warmup_steps=args.max_warmup_steps,
                max_cached_steps=args.max_cached_steps,
                max_continuous_cached_steps=args.max_continuous_cached_steps,
                residual_diff_threshold=args.rdt,
                # NOTE: num_inference_steps is required for Transformer-only interface
                num_inference_steps=28 if args.steps is None else args.steps,
            )
            if args.cache
            else None
        ),
        params_modifiers=[
            ParamsModifier(
                # transformer_blocks
                cache_config=DBCacheConfig().reset(
                    residual_diff_threshold=args.rdt,
                ),
            ),
            ParamsModifier(
                # single_transformer_blocks
                cache_config=DBCacheConfig().reset(
                    residual_diff_threshold=args.rdt * 3,
                ),
            ),
        ],
    )


def run_pipe():
    image = pipe(
        "A cat holding a sign that says hello world",
        num_inference_steps=28 if args.steps is None else args.steps,
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
