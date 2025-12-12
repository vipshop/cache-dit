import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import OvisImagePipeline, OvisImageTransformer2DModel
from utils import (
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
    MemoryTracker,
    create_profiler_from_args,
)
import cache_dit


args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

args = get_args()
print(args)

pipe = OvisImagePipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "OVIS_IMAGE_DIR",
            "AIDC-AI/Ovis-Image-7B",
        )
    ),
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

assert isinstance(pipe.transformer, OvisImageTransformer2DModel)
if args.quantize:
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
        quant_type=args.quantize_type,
        exclude_layers=[
            "embedder",
            "embed",
        ],
    )
    pipe.text_encoder = cache_dit.quantize(
        pipe.text_encoder,
        quant_type=args.quantize_type,
    )

pipe.to("cuda")


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

pipe.set_progress_bar_config(disable=rank != 0)

prompt = 'A creative 3D artistic render where the text "OVIS-IMAGE" is written in a bold, expressive handwritten brush style using thick, wet oil paint. The paint is a mix of vibrant rainbow colors (red, blue, yellow) swirling together like toothpaste or impasto art. You can see the ridges of the brush bristles and the glossy, wet texture of the paint. The background is a clean artist\'s canvas. Dynamic lighting creates soft shadows behind the floating paint strokes. Colorful, expressive, tactile texture, 4k detail.'
if args.prompt is not None:
    prompt = args.prompt


def run_pipe():
    steps = args.steps if args.steps is not None else 28
    if args.profile and args.steps is None:
        steps = 3
    image = pipe(
        prompt,
        negative_prompt="",
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=steps,
        guidance_scale=5.0,  # has separate cfg for ovis image
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


# warmup
_ = run_pipe()

memory_tracker = MemoryTracker() if args.track_memory else None

if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
if args.profile:
    profiler = create_profiler_from_args(args, profile_name="ovis_image_inference")
    with profiler:
        image = run_pipe()
    print(f"Profiler traces saved to: {profiler.output_dir}/{profiler.trace_path.name}")
else:
    image = run_pipe()
end = time.time()


if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

if rank == 0:

    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"ovis_image.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
