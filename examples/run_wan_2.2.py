import os
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
    from cache_dit import ForwardPattern, BlockAdapter

    cache_dit.enable_cache(
        # Only cache for low-noise transformer (occupancy most timesteps)
        # boundary_ratio: 0.875, boundary_timestep=0.875*1000=875
        # t >= boundary_timestep: transformer, high-noise
        # t < boundary_timestep: transformer_2, low-noise
        BlockAdapter(
            pipe=pipe,
            transformer=pipe.transformer_2,
            blocks=pipe.transformer_2.blocks,
            blocks_name="blocks",
        ),
        forward_pattern=ForwardPattern.Pattern_2,
        # Cache context kwargs
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=2,
        max_cached_steps=20,
        max_continuous_cached_steps=2,
        residual_diff_threshold=0.08,
        do_separate_cfg=True,
        enable_taylorseer=True,
        enable_encoder_taylorseer=True,
        taylorseer_order=2,
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

if args.fp8:
    from utils import quantize_fp8

    print("Enable FP8 Quntization for Wan2.2")
    # ensure bfloat16
    pipe.transformer = quantize_fp8(pipe.transformer)
    pipe.transformer_2 = quantize_fp8(pipe.transformer_2)

if args.compile:
    cache_dit.set_compile_configs()
    if not args.fp8:
        pipe.transformer.compile_repeated_blocks(fullgraph=True)
        pipe.transformer_2.compile_repeated_blocks(fullgraph=True)
    else:
        pipe.transformer = torch.compile(pipe.transformer)
        pipe.transformer_2 = torch.compile(pipe.transformer_2)

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

stats = cache_dit.summary(
    pipe.transformer_2,
    details=True,
)

time_cost = end - start
save_path = (
    f"wan2.2.C{int(args.compile)}_Q{int(args.fp8)}_"
    f"{cache_dit.strify(stats)}.mp4"
)
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=16)
