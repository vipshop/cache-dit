import os
import sys

sys.path.append("..")

import time
import torch
import diffusers
from diffusers import (
    AutoencoderKLWan,
    WanTransformer3DModel,
    WanImageToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from utils import get_args, GiB, strify, cachify
import cache_dit
import numpy as np

args = get_args()
print(args)

pipe = WanImageToVideoPipeline.from_pretrained(
    os.environ.get(
        "WAN_2_2_I2V_DIR",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
)

# image = load_image("./img.png")
image = load_image("../data/flf2v_input_first_frame.png")

max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = (
    pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
)
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
    # Use the UniPCMultistepScheduler with the specified flow shift
    flow_shift = 3.0 if height == 480 else 5.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )

if args.cache:
    from cache_dit import (
        ForwardPattern,
        BlockAdapter,
        ParamsModifier,
        DBCacheConfig,
    )

    cachify(
        args,
        BlockAdapter(
            pipe=pipe,
            transformer=[
                pipe.transformer,
                pipe.transformer_2,
            ],
            blocks=[
                pipe.transformer.blocks,
                pipe.transformer_2.blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_2,
                ForwardPattern.Pattern_2,
            ],
            params_modifiers=[
                # high-noise transformer only have 30% steps
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=4,
                        max_cached_steps=8,
                    ),
                ),
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=2,
                        max_cached_steps=20,
                    ),
                ),
            ],
            has_separate_cfg=True,
        ),
    )

# Wan currently requires installing diffusers from source
assert isinstance(pipe.vae, AutoencoderKLWan)  # enable type check for IDE
if diffusers.__version__ >= "0.34.0":
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
else:
    print(
        "Wan pipeline requires diffusers version >= 0.34.0 "
        "for vae tiling and slicing, please install diffusers "
        "from source."
    )

assert isinstance(pipe.transformer, WanTransformer3DModel)
assert isinstance(pipe.transformer_2, WanTransformer3DModel)

if args.quantize:
    assert isinstance(args.quantize_type, str)
    if args.quantize_type.endswith("wo"):  # weight only
        pipe.transformer = cache_dit.quantize(
            pipe.transformer,
            quant_type=args.quantize_type,
        )
    # We only apply activation quantization (default: FP8 DQ)
    # for low-noise transformer to avoid non-trivial precision
    # downgrade.
    pipe.transformer_2 = cache_dit.quantize(
        pipe.transformer_2,
        quant_type=args.quantize_type,
    )


def run_pipe():
    video = pipe(
        prompt=(
            "镜头先聚焦到脚步，然后脚踢到石头，然后人往前摔倒，手中的素描本向上方飞出，镜头瞬间聚焦至素描本，随着素描本飞向远方"
        ),
        height=height,
        width=width,
        image=image,
        num_frames=81,
        num_inference_steps=50,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return video


if args.compile or args.quantize:
    cache_dit.set_compile_configs()
    pipe.transformer.compile_repeated_blocks(fullgraph=True)
    pipe.transformer_2.compile_repeated_blocks(fullgraph=True)

    # warmup
    run_pipe()

start = time.time()
video = run_pipe()
end = time.time()

cache_dit.summary(pipe, details=True)

time_cost = end - start
save_path = f"wan2.2-i2v.{strify(args, pipe)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=16)
