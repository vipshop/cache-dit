import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
    PipelineQuantizationConfig,
)
from utils import (
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
    MemoryTracker,
    print_rank0,
)
import cache_dit


args = get_args()
rank, device = maybe_init_distributed(args)
print_rank0(args)

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
    cachify(args, pipe)

assert isinstance(pipe.transformer, FluxTransformer2DModel)

pipe.set_progress_bar_config(disable=rank != 0)

# Set default prompt
prompt = "A cat holding a sign that says hello world"
if args.prompt is not None:
    prompt = args.prompt


def run_pipe(pipe: FluxPipeline):
    image = pipe(
        prompt,
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=28 if args.steps is None else args.steps,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

# warmup
_ = run_pipe(pipe)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe(pipe)
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report(rank)

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
