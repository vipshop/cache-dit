import os
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

pipe = MochiPipeline.from_pretrained(
    os.environ.get(
        "MOCHI_DIR",
        "genmo/mochi-1-preview",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B8, good balance between performance and precision
cache_options = CacheType.default_options(CacheType.DBCache)

apply_cache_on_pipe(pipe, **cache_options)

pipe.enable_vae_tiling()

prompt = (
    "Close-up of a chameleon's eye, with its scaly skin "
    "changing color. Ultra high resolution 4k."
)
video = pipe(
    prompt,
    num_frames=84,
).frames[0]

print("Saving video to mochi.mp4")
export_to_video(video, "mochi.mp4", fps=30)
