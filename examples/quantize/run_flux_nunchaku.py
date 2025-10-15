import os
import sys

sys.path.append("..")
import time

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel

from nunchaku.models.transformers.transformer_flux_v2 import (
    NunchakuFluxTransformer2DModelV2,
)
from utils import get_args, strify
import cache_dit

args = get_args()
print(args)

nunchaku_flux_dir = os.environ.get(
    "NUNCHAKA_FLUX_DIR",
    "nunchaku-tech/nunchaku-flux.1-dev",
)
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"{nunchaku_flux_dir}/svdq-int4_r32-flux.1-dev.safetensors",
)
pipe: FluxPipeline = FluxPipeline.from_pretrained(
    os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
    transformer=transformer,
    torch_dtype=torch.bfloat16,
).to("cuda")


if args.cache:
    from cache_dit import (
        ParamsModifier,
        DBCacheConfig,
        TaylorSeerCalibratorConfig,
    )

    cache_dit.enable_cache(
        pipe,
        cache_config=DBCacheConfig(
            Fn_compute_blocks=args.Fn,
            Bn_compute_blocks=args.Bn,
            max_warmup_steps=args.max_warmup_steps,
            max_cached_steps=args.max_cached_steps,
            max_continuous_cached_steps=args.max_continuous_cached_steps,
            residual_diff_threshold=args.rdt,
        ),
        calibrator_config=(
            TaylorSeerCalibratorConfig(
                taylorseer_order=args.taylorseer_order,
            )
            if args.taylorseer
            else None
        ),
        params_modifiers=[
            ParamsModifier(
                # transformer_blocks
                cache_config=DBCacheConfig().reset(
                    residual_diff_threshold=args.rdt
                ),
            ),
            ParamsModifier(
                # single_transformer_blocks
                cache_config=DBCacheConfig().reset(
                    residual_diff_threshold=args.rdt * 3
                ),
            ),
        ],
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
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

    # warmup
    _ = run_pipe(pipe)


start = time.time()
image = run_pipe(pipe)
end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.nunchaku.int4.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
