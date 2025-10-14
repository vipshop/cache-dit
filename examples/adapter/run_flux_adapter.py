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

    from cache_dit import (
        ForwardPattern,
        BlockAdapter,
        ParamsModifier,
        DBCacheConfig,
    )
    from cache_dit.utils import is_diffusers_at_least_0_3_5
    from diffusers import FluxTransformer2DModel

    assert isinstance(pipe.transformer, FluxTransformer2DModel)

    if is_diffusers_at_least_0_3_5():
        # For diffusers >= 0.35.0
        cache_dit.enable_cache(
            BlockAdapter(
                pipe=pipe,
                transformer=pipe.transformer,
                blocks=[
                    pipe.transformer.transformer_blocks,
                    pipe.transformer.single_transformer_blocks,
                ],
                forward_pattern=[
                    ForwardPattern.Pattern_1,
                    ForwardPattern.Pattern_1,
                ],
                params_modifiers=[
                    ParamsModifier(
                        cache_config=DBCacheConfig(
                            residual_diff_threshold=0.12,
                        ),
                    ),
                    ParamsModifier(
                        cache_config=DBCacheConfig(
                            Fn_compute_blocks=1,
                            residual_diff_threshold=0.25,
                        ),
                    ),
                ],
            ),
        )

    else:

        # For diffusers <= 0.34.0
        cache_dit.enable_cache(
            BlockAdapter(
                pipe=pipe,
                transformer=pipe.transformer,
                blocks=[
                    pipe.transformer.transformer_blocks,
                    pipe.transformer.single_transformer_blocks,
                ],
                forward_pattern=[
                    ForwardPattern.Pattern_1,
                    ForwardPattern.Pattern_3,
                ],
                params_modifiers=[
                    ParamsModifier(
                        cache_config=DBCacheConfig(
                            residual_diff_threshold=0.12,
                        ),
                    ),
                    ParamsModifier(
                        cache_config=DBCacheConfig(
                            Fn_compute_blocks=1,
                            residual_diff_threshold=0.25,
                        ),
                    ),
                ],
            ),
        )


start = time.time()
image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.adapter.{cache_dit.strify(pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
