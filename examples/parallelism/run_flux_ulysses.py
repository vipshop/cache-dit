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


args = get_args()
print(args)

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

pipe.transformer.enable_parallelism(
    config=ContextParallelConfig(ulysses_degree=dist.get_world_size()),
)

start = time.time()
image = pipe(
    "A cat holding a sign that says hello world",
    num_inference_steps=28,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]

end = time.time()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux.ulysses{dist.get_world_size()}.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

if dist.is_initialized():
    dist.destroy_process_group()
