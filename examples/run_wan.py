import os
import time
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

cache_options = {
    "cache_type": CacheType.DBCache,
    "warmup_steps": 0,
    "max_cached_steps": -1,  # -1 means no limit
    # Fn=1, Bn=0, means FB Cache, otherwise, Dual Block Cache
    "Fn_compute_blocks": 8,  # Fn, F8, etc.
    "Bn_compute_blocks": 8,  # Bn, B16, etc.
    "non_compute_blocks_diff_threshold": 0.08,
    "residual_diff_threshold": 0.08,
    "enable_alter_cache": False,
    # releative token diff threshold, default is 0.0
    "important_condition_threshold": 0.00,
    # TaylorSeer options
    "enable_taylorseer": False,
    "enable_encoder_taylorseer": False,
    # Taylorseer cache type cache be hidden_states or residual
    "taylorseer_cache_type": "residual",
    "taylorseer_kwargs": {
        "n_derivatives": 2,
    },
}

# Default options, F8B8, good balance between performance and precision
apply_cache_on_pipe(pipe, **cache_options)

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

start = time.time()
video = pipe(
    prompt=(
        "An astronaut dancing vigorously on the moon with earth "
        "flying past in the background, hyperrealistic"
    ),
    negative_prompt="",
    height=height,
    width=width,
    num_frames=49,
    num_inference_steps=35,
).frames[0]
end = time.time()

if hasattr(pipe.transformer, "_cached_steps"):
    cached_steps = pipe.transformer._cached_steps
    print(f"Cache steps: {len(cached_steps)}, {cached_steps} ")

time_cost = end - start
print(f"Time cost: {time_cost:.2f}s")
print("Saving video to wan.mp4")
export_to_video(video, "wan.0.mp4", fps=16)
