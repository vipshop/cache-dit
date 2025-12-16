import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
)
from utils import (
    get_args,
    strify,
    maybe_apply_optimization,
    maybe_init_distributed,
    maybe_destroy_distributed,
    create_profiler_from_args,
    pipe_quant_bnb_4bit_config,
)
import cache_dit


args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

pipe: FluxPipeline = FluxPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "FLUX_DIR",
            "black-forest-labs/FLUX.1-dev",
        )
    ),
    torch_dtype=torch.bfloat16,
    quantization_config=pipe_quant_bnb_4bit_config(
        args,
        components_to_quantize=["text_encoder_2"],
    ),
).to("cuda")

maybe_apply_optimization(args, pipe)

assert isinstance(pipe.transformer, FluxTransformer2DModel)

pipe.set_progress_bar_config(disable=rank != 0)

# Set default prompt
prompt = "A cat holding a sign that says hello world"
if args.prompt is not None:
    prompt = args.prompt


height = 1024 if args.height is None else args.height
width = 1024 if args.width is None else args.width


def run_pipe(pipe: FluxPipeline):
    steps = 28 if args.steps is None else args.steps
    if args.profile and args.steps is None:
        steps = 3
    image = pipe(
        prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


# warmup
_ = run_pipe(pipe)

start = time.time()
if args.profile:
    profiler = create_profiler_from_args(args, profile_name="flux_cp_inference")
    with profiler:
        image = run_pipe(pipe)
    if rank == 0:
        print(f"Profiler traces saved to: {profiler.output_dir}/{profiler.trace_path.name}")
else:
    image = run_pipe(pipe)
end = time.time()


if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux.{height}x{width}.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
