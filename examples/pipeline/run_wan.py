import os
import sys

sys.path.append("..")

import time
import torch
import diffusers
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)


height, width = 480, 832
pipe = WanPipeline.from_pretrained(
    os.environ.get(
        "WAN_DIR",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",  # "num_layers": 30,
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


if args.cache:
    cachify(args, pipe)

# Enable memory savings
pipe.enable_model_cpu_offload()

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

prompt = (
    "An astronaut dancing vigorously on the moon with earth "
    "flying past in the background, hyperrealistic"
)
if args.prompt is not None:
    prompt = args.prompt

negative_prompt = ""
if args.negative_prompt is not None:
    negative_prompt = args.negative_prompt

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    num_frames=49,
    num_inference_steps=35,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"wan.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=16)
