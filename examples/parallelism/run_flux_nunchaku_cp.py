import os
import sys

sys.path.append("..")
import time

import torch
import torch.distributed as dist
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    PipelineQuantizationConfig,
)
from nunchaku.models.transformers.transformer_flux_v2 import (
    NunchakuFluxTransformer2DModelV2,
)
from utils import (
    get_args,
    strify,
    maybe_init_distributed,
    maybe_destroy_distributed,
)
import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

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
    quantization_config=(
        PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["text_encoder_2"],
        )
        if args.quantize
        else None
    ),
).to("cuda")


if args.cache or args.parallel_type is not None:
    from cache_dit import (
        ParamsModifier,
        DBCacheConfig,
        TaylorSeerCalibratorConfig,
        ParallelismConfig,
    )

    cache_dit.enable_cache(
        pipe,
        cache_config=(
            DBCacheConfig(
                Fn_compute_blocks=args.Fn,
                Bn_compute_blocks=args.Bn,
                max_warmup_steps=args.max_warmup_steps,
                max_cached_steps=args.max_cached_steps,
                max_continuous_cached_steps=args.max_continuous_cached_steps,
                residual_diff_threshold=args.rdt,
            )
            if args.cache
            else None
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
        # In order to enable parallelism for nunchaku flux transformer,
        # please use our modified fork: https://github.com/vipshop/nunchaku
        parallelism_config=(
            ParallelismConfig(
                ulysses_size=(
                    dist.get_world_size()
                    if args.parallel_type == "ulysses"
                    else None
                ),
                ring_size=(
                    dist.get_world_size()
                    if args.parallel_type == "ring"
                    else None
                ),
            )
            if args.parallel_type in ["ulysses", "ring"]
            else None
        ),
    )

    if args.parallel_type in ["ulysses", "ring"]:
        assert isinstance(pipe.transformer, NunchakuFluxTransformer2DModelV2)
        pipe.transformer.set_native_parallel_flag(True)

assert isinstance(pipe.transformer, FluxTransformer2DModel)

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(pipe: FluxPipeline):
    image = pipe(
        "A cat holding a sign that says hello world",
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=28 if args.steps is None else args.steps,
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

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux.nunchaku.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)


maybe_destroy_distributed()
