# Adapted from: https://github.com/chengzeyi/ParaAttention/blob/main/first_block_cache_examples/run_hunyuan_video.py
import os
import torch
from diffusers.utils import export_to_video
from diffusers import (
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
    AutoencoderKLHunyuanVideo,
)
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

model_id = os.environ.get("HUNYUAN_DIR", "tencent/HunyuanVideo")


def get_gpu_memory_in_gib():
    if not torch.cuda.is_available():
        return 0

    try:
        total_memory_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device(),
        ).total_memory
        total_memory_gib = total_memory_bytes / (1024**3)
        return int(total_memory_gib)
    except Exception:
        return 0


transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
).to("cuda")


# Default options, F8B8, good balance between performance and precision
apply_cache_on_pipe(pipe, **CacheType.default_options(CacheType.DBCache))

assert isinstance(
    pipe.vae, AutoencoderKLHunyuanVideo
)  # enable type check for IDE

# Enable memory savings
pipe.enable_model_cpu_offload()
if get_gpu_memory_in_gib() <= 48:
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


output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=30,
).frames[0]

print("Saving video to hunyuan_video.mp4")
export_to_video(output, "hunyuan_video.mp4", fps=15)
