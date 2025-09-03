import os
import sys

sys.path.append("..")

import time
import torch
from diffusers.utils import export_to_video
from diffusers import (
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
    AutoencoderKLHunyuanVideo,
)
from utils import GiB, get_args
import cache_dit


args = get_args()
print(args)


model_id = os.environ.get("HUNYUAN_DIR", "tencent/HunyuanVideo")
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")


if args.cache:
    cache_dit.enable_cache(pipe)


assert isinstance(
    pipe.vae, AutoencoderKLHunyuanVideo
)  # enable type check for IDE

# Enable memory savings
pipe.enable_model_cpu_offload()
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


start = time.time()
output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=30,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"hunyuan_video.{cache_dit.strify(stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(output, save_path, fps=15)
