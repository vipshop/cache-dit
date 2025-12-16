import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel

from utils import (
    GiB,
    maybe_apply_optimization,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    pipe_quant_bnb_4bit_config,
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
    quantization_config=pipe_quant_bnb_4bit_config(
        args,
        components_to_quantize=["text_encoder", "transformer"],
    ),
)


from cache_dit import DBCacheConfig, ParamsModifier

maybe_apply_optimization(
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


# warmup
_ = run_pipe(warmup=True)

start = time.time()
image = run_pipe()
end = time.time()


if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"flux2.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
