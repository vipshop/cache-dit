import os
import sys

sys.path.append("..")

import time
import requests
from io import BytesIO

import torch
from PIL import Image
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from utils import (
    MemoryTracker,
    GiB,
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
    from cache_dit import DBCacheConfig, ParamsModifier

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
        params_modifiers=[
            ParamsModifier(
                # Modified config only for transformer_blocks
                # Must call the `reset` method of DBCacheConfig.
                cache_config=DBCacheConfig().reset(
                    residual_diff_threshold=args.rdt,
                ),
            ),
            ParamsModifier(
                # Modified config only for single_transformer_blocks
                # NOTE: FLUX.2, single_transformer_blocks should have `higher`
                # residual_diff_threshold because of the precision error
                # accumulation from previous transformer_blocks
                cache_config=DBCacheConfig().reset(
                    residual_diff_threshold=args.rdt * 3,
                ),
            ),
        ],
    )

torch.cuda.empty_cache()

world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

if world_size < 4 and GiB() <= 48:
    assert not args.compile, "Compilation requires more GPU memory. Please disable it."
    if world_size < 2:
        pipe.enable_sequential_cpu_offload(device=device)
        print("Enabled sequential CPU offload.")
    else:
        pipe.enable_model_cpu_offload(device=device)
        print("Enabled model CPU offload.")
else:
    pipe.to(device)

assert isinstance(pipe.transformer, Flux2Transformer2DModel)

pipe.set_progress_bar_config(disable=rank != 0)

# Image editing prompt (from user example)
prompt = "Put a birthday hat on the dog in the image"

if args.prompt is not None:
    prompt = args.prompt

# Download input image
image_url = "https://modelscope.oss-cn-beijing.aliyuncs.com/Dog.png"
if args.image_path is not None:
    # Support local image path
    if os.path.exists(args.image_path):
        input_image = Image.open(args.image_path).convert("RGB")
    else:
        # Try to download as URL
        image_url = args.image_path
        response = requests.get(image_url)
        input_image = Image.open(BytesIO(response.content)).convert("RGB")
else:
    # Use default example image
    response = requests.get(image_url)
    input_image = Image.open(BytesIO(response.content)).convert("RGB")

if rank == 0:
    print(f"Input image loaded: {input_image.size}")
    print(f"Prompt: {prompt}")


def run_pipe(warmup: bool = False):
    generator = torch.Generator("cpu").manual_seed(42)
    image = pipe(
        prompt=prompt,
        image=input_image,  # Pass input image for editing
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
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
    print(f"Time cost: {time_cost:.2f}s")
    
    save_path = f"flux2_edit.{strify(args, pipe)}.png"
    print(f"Saving edited image to {save_path}")
    image.save(save_path)
    print(f"Image saved successfully!")

maybe_destroy_distributed()

