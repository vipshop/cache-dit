import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import OvisImagePipeline
from utils import (
    get_args,
    maybe_apply_optimization,
    maybe_init_distributed,
    maybe_destroy_distributed,
    create_profiler_from_args,
    pipe_quant_bnb_4bit_config,
)


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
    quantization_config=pipe_quant_bnb_4bit_config(
        args,
        components_to_quantize=["text_encoder", "transformer"],
    ),
)

maybe_apply_optimization(args, pipe)


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


start = time.time()
if args.profile:
    profiler = create_profiler_from_args(args, profile_name="ovis_image_inference")
    with profiler:
        image = run_pipe()
    print(f"Profiler traces saved to: {profiler.output_dir}/{profiler.trace_path.name}")
else:
    image = run_pipe()
end = time.time()
time_cost = end - start

maybe_destroy_distributed(args, pipe, "ovis.image", time_cost=time_cost, image=image)
