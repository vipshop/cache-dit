import os
import sys

sys.path.append("..")

import time
import torch
import diffusers
from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from utils import get_args, GiB
import cache_dit


args = get_args()
print(args)


height, width = 480, 832
pipe = WanPipeline.from_pretrained(
    os.environ.get(
        "WAN_2_2_DIR",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
)

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
    # Use the UniPCMultistepScheduler with the specified flow shift
    flow_shift = 3.0 if height == 480 else 5.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )


if args.cache:
    from cache_dit import ForwardPattern, BlockAdapter, ParamsModifier

    cache_dit.enable_cache(
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
            blocks_name=[
                "blocks",
                "blocks",
            ],
            forward_pattern=[
                ForwardPattern.Pattern_2,
                ForwardPattern.Pattern_2,
            ],
            params_modifiers=[
                # high-noise transformer only have 30% steps
                ParamsModifier(
                    max_cached_steps=8,
                ),
                ParamsModifier(
                    max_cached_steps=25,
                ),
            ],
            has_separate_cfg=True,
        ),
        # Common cache params
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=2,
        max_continuous_cached_steps=2,
        residual_diff_threshold=0.12,
        enable_taylorseer=True,
        enable_encoder_taylorseer=True,
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

if args.compile or args.quantize:
    cache_dit.set_compile_configs()
    pipe.transformer.compile_repeated_blocks(fullgraph=True)
    pipe.transformer_2.compile_repeated_blocks(fullgraph=True)

    # warmup
    video = pipe(
        prompt=(
            "An astronaut dancing vigorously on the moon with earth "
            "flying past in the background, hyperrealistic"
        ),
        height=height,
        width=width,
        num_frames=81,
        num_inference_steps=50,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]


start = time.time()
video = pipe(
    prompt=(
        "An astronaut dancing vigorously on the moon with earth "
        "flying past in the background, hyperrealistic"
    ),
    negative_prompt="",
    height=height,
    width=width,
    num_frames=81,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

cache_dit.summary(pipe.transformer, details=True)
cache_dit.summary(pipe.transformer_2, details=True)

time_cost = end - start
save_path = (
    f"wan2.2.C{int(args.compile)}_Q{int(args.quantize)}"
    f"{'' if not args.quantize else ('_' + args.quantize_type)}_"
    f"{cache_dit.strify(pipe)}.mp4"
)
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=16)
