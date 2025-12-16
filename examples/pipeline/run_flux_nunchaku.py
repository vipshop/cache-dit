import os
import sys

sys.path.append("..")
import time

import torch
from diffusers import (
    FluxPipeline,
    FluxTransformer2DModel,
)
from nunchaku.models.transformers.transformer_flux_v2 import (
    NunchakuFluxTransformer2DModelV2,
)
from utils import (
    get_args,
    maybe_apply_optimization,
    maybe_init_distributed,
    maybe_destroy_distributed,
    pipe_quant_bnb_4bit_config,
)

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

nunchaku_flux_dir = os.environ.get(
    "NUNCHAKA_FLUX_DIR",
    "nunchaku-tech/nunchaku-flux.1-dev",
)
transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
    f"{nunchaku_flux_dir}/svdq-int4_r32-flux.1-dev.safetensors",
)
pipe: FluxPipeline = FluxPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get("FLUX_DIR", "black-forest-labs/FLUX.1-dev")
    ),
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    quantization_config=pipe_quant_bnb_4bit_config(
        args,
        components_to_quantize=["text_encoder_2"],
    ),
).to("cuda")


from cache_dit import ParamsModifier, DBCacheConfig

maybe_apply_optimization(
    pipe,
    params_modifiers=[
        ParamsModifier(
            # transformer_blocks
            cache_config=DBCacheConfig().reset(residual_diff_threshold=args.rdt),
        ),
        ParamsModifier(
            # single_transformer_blocks
            cache_config=DBCacheConfig().reset(residual_diff_threshold=args.rdt * 3),
        ),
    ],
)

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


# warmup
_ = run_pipe(pipe)

start = time.time()
image = run_pipe(pipe)
end = time.time()
time_cost = end - start

maybe_destroy_distributed(args, pipe, "flux.nunchaku", time_cost=time_cost, image=image)
