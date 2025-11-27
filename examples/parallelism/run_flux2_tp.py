import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from utils import (
    MemoryTracker,
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
)

import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

pipe: Flux2Pipeline = Flux2Pipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "FLUX_2_DIR",
            "black-forest-labs/FLUX.2-dev",
        )
    ),
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(
        args,
        pipe,
        extra_parallel_modules=(
            # Specify extra modules to be parallelized in addition to the main transformer,
            # e.g., text_encoder_2 in FluxPipeline, text_encoder in Flux2Pipeline. Currently,
            # only supported in native pytorch backend (namely, Tensor Parallelism).
            [pipe.text_encoder]
            if args.parallel_type == "tp"
            else []
        ),
    )

# tp_mesh: DeviceMesh = init_device_mesh(
#     device_type="cuda",
#     mesh_shape=[torch.distributed.get_world_size()],
# )
# tp_planer = Flux2TensorParallelismPlanner()
# tp_planer.parallelize_text_encoder(
#     text_encoder=pipe.text_encoder,
#     tp_mesh=tp_mesh,
# )
# pipe.text_encoder.to("cpu")
# torch.cuda.empty_cache()

# tp_planer.parallelize_transformer(
#     transformer=pipe.transformer,
#     tp_mesh=tp_mesh,
# )
# pipe.transformer.to("cpu")
torch.cuda.empty_cache()

pipe.enable_model_cpu_offload(device=device)

assert isinstance(pipe.transformer, Flux2Transformer2DModel)

pipe.set_progress_bar_config(disable=rank != 0)

prompt = (
    "Realistic macro photograph of a hermit crab using a soda can as its shell, "
    "partially emerging from the can, captured with sharp detail and natural colors, "
    "on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean "
    "waves in the background. The can has the text `BFL Diffusers` on it and it has a color "
    "gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
)

if args.prompt is not None:
    prompt = args.prompt


def run_pipe(warmup: bool = False):
    generator = torch.Generator("cpu").manual_seed(42)
    image = pipe(
        prompt=prompt,
        # 28 steps can be a good trade-off
        num_inference_steps=5 if warmup else (50 if args.steps is None else args.steps),
        guidance_scale=4,
        generator=generator,
    ).images[0]
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

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
    memory_tracker.report()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux2.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
