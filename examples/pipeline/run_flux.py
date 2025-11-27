import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from utils import get_args, strify, cachify, MemoryTracker, create_profiler_from_args
import cache_dit


args = get_args()
print(args)


pipe = FluxPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "FLUX_DIR",
            "black-forest-labs/FLUX.1-dev",
        )
    ),
    torch_dtype=torch.bfloat16,
)

if args.cache:
    cachify(args, pipe)

assert isinstance(pipe.transformer, FluxTransformer2DModel)
if args.quantize:
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
        quant_type=args.quantize_type,
        exclude_layers=[
            "embedder",
            "embed",
        ],
    )
    pipe.text_encoder_2 = cache_dit.quantize(
        pipe.text_encoder_2,
        quant_type=args.quantize_type,
    )
    print(f"Applied quantization: {args.quantize_type} to Transformer and Text Encoder 2.")

pipe.to("cuda")

if args.attn is not None:
    if hasattr(pipe.transformer, "set_attention_backend"):
        pipe.transformer.set_attention_backend(args.attn)
        print(f"Set attention backend to {args.attn}")

if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)
    pipe.text_encoder = torch.compile(pipe.text_encoder)
    pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2)
    pipe.vae = torch.compile(pipe.vae)

# Set default prompt
prompt = "A cat holding a sign that says hello world"
if args.prompt is not None:
    prompt = args.prompt


def run_pipe():
    steps = args.steps if args.steps is not None else 28
    if args.profile and args.steps is None:
        steps = 3
    image = pipe(
        prompt,
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=steps,
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
    profiler = create_profiler_from_args(args, profile_name="flux_inference")
    with profiler:
        image = run_pipe()
    print(f"Profiler traces saved to: {profiler.output_dir}/{profiler.trace_path.name}")
else:
    image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
