import os
import sys

sys.path.append("..")

import time
import torch
from diffusers.utils import export_to_video
from diffusers import HunyuanVideoPipeline, AutoencoderKLHunyuanVideo
from utils import GiB, get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)

model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get("HUNYUAN_VIDEO_DIR", "hunyuanvideo-community/HunyuanVideo")
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=("balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None),
)


if args.cache:
    cachify(args, pipe)

# When device_map is None, we need to explicitly move the model to GPU
# or enable CPU offload to avoid running on CPU
if torch.cuda.device_count() <= 1:
    # Single GPU: use CPU offload for memory efficiency
    pipe.enable_model_cpu_offload()
elif torch.cuda.device_count() > 1 and pipe.device.type == "cpu":
    # Multi-GPU but model is on CPU (device_map was None): move to default GPU
    pipe.to("cuda")

assert isinstance(pipe.vae, AutoencoderKLHunyuanVideo)

# Enable memory savings
if GiB() <= 48:
    pipe.vae.enable_tiling(
        # Make it runnable on GPUs with 48GB memory
        tile_sample_min_height=128,
        tile_sample_stride_height=96,
        tile_sample_min_width=128,
        tile_sample_stride_width=96,
        tile_sample_min_num_frames=32,
        tile_sample_stride_num_frames=24,
    )
else:
    pipe.vae.enable_tiling()

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."
if args.prompt is not None:
    prompt = args.prompt

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
output = pipe(
    prompt=prompt,
    num_frames=18,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"hunyuan_video.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(output, save_path, fps=9)
