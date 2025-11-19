import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from utils import get_args, strify, cachify
import cache_dit


args = get_args()
print(args)


pipe = FluxPipeline.from_pretrained(
    os.environ.get(
        "FLUX_DIR",
        "black-forest-labs/FLUX.1-dev",
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
            "norm_out",
            "proj_out",
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


def run_pipe():
    image = pipe(
        "A cat holding a sign that says hello world",
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        num_inference_steps=28 if args.steps is None else args.steps,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)
    if args.quantize:
        pipe.text_encoder_2 = torch.compile(pipe.text_encoder_2)


# warmup
_ = run_pipe()

start = time.time()
image = run_pipe()
end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"flux.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
