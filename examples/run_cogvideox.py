import os
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

pipe = CogVideoXPipeline.from_pretrained(
    os.environ.get(
        "COGVIDEOX_DIR",
        "THUDM/CogVideoX-5b",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B8, good balance between performance and precision
cache_options = CacheType.default_options(CacheType.DBCache)

apply_cache_on_pipe(pipe, **cache_options)

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, "
    "sits on a wooden stool in a serene bamboo forest. The "
    "panda's fluffy paws strum a miniature acoustic guitar, "
    "producing soft, melodic tunes. Nearby, a few other pandas "
    "gather, watching curiously and some clapping in rhythm. "
    "Sunlight filters through the tall bamboo, casting a gentle "
    "glow on the scene. The panda's face is expressive, showing "
    "concentration and joy as it plays. The background includes "
    "a small, flowing stream and vibrant green foliage, enhancing "
    "the peaceful and magical atmosphere of this unique musical "
    "performance."
)
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator("cuda").manual_seed(0),
).frames[0]

print("Saving video to cogvideox.mp4")
export_to_video(video, "cogvideox.mp4", fps=8)
