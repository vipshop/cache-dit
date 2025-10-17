import os
import sys

sys.path.append("..")
import time

import torch
import torch.distributed as dist
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    ContextParallelConfig,
)
from nunchaku.models.transformers.transformer_flux_v2 import (
    NunchakuFluxTransformer2DModelV2,
)
from utils import get_args, strify
import cache_dit

parser = get_args(parse=False)
parser.add_argument(
    "--parallel-type",
    type=str,
    default="none",
    choices=["ulysses", "ring", "none"],
)
args = parser.parse_args()
print(args)

if args.parallel_type != "none":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

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

assert isinstance(pipe.transformer, FluxTransformer2DModel)

if args.parallel_type != "none":
    # Now only _native_cudnn is supported for parallelism
    # issue: https://github.com/huggingface/diffusers/pull/12443
    pipe.transformer.set_attention_backend("_native_cudnn")

if args.parallel_type == "ulysses":
    pipe.transformer.enable_parallelism(
        config=ContextParallelConfig(ulysses_degree=dist.get_world_size()),
    )
elif args.parallel_type == "ring":
    pipe.transformer.enable_parallelism(
        config=ContextParallelConfig(ring_degree=dist.get_world_size()),
    )
else:
    print("No parallelism is enabled.")


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

if args.parallel_type != "none":
    if rank == 0:
        cache_dit.summary(pipe)

        time_cost = end - start
        save_path = (
            f"flux.nunchaku.{args.parallel_type}"
            f"{dist.get_world_size()}.{strify(args, pipe)}.png"
        )
        print(f"Time cost: {time_cost:.2f}s")
        print(f"Saving image to {save_path}")
        image.save(save_path)
else:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux.nunchaku.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

if dist.is_initialized():
    dist.destroy_process_group()
