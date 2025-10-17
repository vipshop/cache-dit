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
from utils import get_args, strify, cachify
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


pipe: FluxPipeline = FluxPipeline.from_pretrained(
    os.environ.get(
        "FLUX_DIR",
        "black-forest-labs/FLUX.1-dev",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    cachify(args, pipe)

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
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

    # warmup
    _ = run_pipe(pipe)


start = time.time()
image = run_pipe(pipe)
end = time.time()

if args.parallel_type != "none":
    if rank == 0:
        cache_dit.summary(pipe)

        time_cost = end - start
        save_path = (
            f"flux.{args.parallel_type}{dist.get_world_size()}."
            f"{strify(args, pipe)}.png"
        )
        print(f"Time cost: {time_cost:.2f}s")
        print(f"Saving image to {save_path}")
        image.save(save_path)
else:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

if dist.is_initialized():
    dist.destroy_process_group()
