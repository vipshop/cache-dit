import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from diffusers.quantizers import PipelineQuantizationConfig

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

if GiB() < 128:
    assert args.quantize, "Quantization is required to fit FLUX.2 in <128GB memory."
    assert args.quantize_type in ["bitsandbytes_4bit", "float8_weight_only"], (
        f"Unsupported quantization type: {args.quantize_type}, only supported"
        "'bitsandbytes_4bit (bnb_4bit)' and 'float8_weight_only'."
    )

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
    quantization_config=(
        (
            PipelineQuantizationConfig(
                quant_backend="bitsandbytes_4bit",
                quant_kwargs={
                    "load_in_4bit": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                },
                # 112/4 = 28GB total for text_encoder + transformer in 4-bit
                components_to_quantize=(
                    ["text_encoder", "transformer"]
                    if not args.parallel_text_encoder
                    else ["transformer"]
                ),
            )
        )
        if args.quantize and args.quantize_type in ("bitsandbytes_4bit",)
        else None
    ),
)


if args.cache or args.parallel_type is not None:
    from cache_dit import DBCacheConfig, ParamsModifier

    cachify(
        args,
        pipe,
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

if args.quantize and args.quantize_type == "float8_weight_only":
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
        quant_type=args.quantize_type,
        exclude_layers=[
            "img_in",
            "txt_in",
        ],
    )
    pipe.text_encoder = cache_dit.quantize(
        pipe.text_encoder,
        quant_type=args.quantize_type,
    )

if args.quantize_type == "bitsandbytes_4bit" or (args.parallel_text_encoder and args.quantize):
    pipe.to(device)
else:
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
    generator = torch.Generator("cpu").manual_seed(0)
    image = pipe(
        prompt=prompt,
        height=1024 if args.height is None else args.height,
        width=1024 if args.width is None else args.width,
        # 28 steps can be a good trade-off
        num_inference_steps=5 if warmup else (28 if args.steps is None else args.steps),
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
