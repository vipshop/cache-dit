import os
import torch
import diffusers
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

height, width = 480, 832
pipe = WanPipeline.from_pretrained(
    os.environ.get(
        "WAN_DIR",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    ),
    torch_dtype=torch.bfloat16,
)

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
    # Use the UniPCMultistepScheduler with the specified flow shift
    flow_shift = 3.0 if height == 480 else 5.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )

pipe.to("cuda")

# Default options, F8B8, good balance between performance and precision
apply_cache_on_pipe(pipe, **CacheType.default_options(CacheType.DBCache))

# Enable memory savings
pipe.enable_model_cpu_offload()

# Wan currently requires installing diffusers from source
assert isinstance(pipe.vae, AutoencoderKLWan)  # enable type check for IDE
if diffusers.__version__ >= "0.34.0.dev0":
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
else:
    print(
        "Wan pipeline requires diffusers version >= 0.34.0.dev0 "
        "for vae tiling and slicing, please install diffusers "
        "from source."
    )


video = pipe(
    prompt=(
        "An astronaut dancing vigorously on the moon with earth "
        "flying past in the background, hyperrealistic"
    ),
    negative_prompt="",
    height=height,
    width=width,
    num_frames=81,
    num_inference_steps=30,
).frames[0]

print("Saving video to wan.mp4")
export_to_video(video, "wan.mp4", fps=15)
