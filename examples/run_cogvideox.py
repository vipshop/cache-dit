import os
import torch
from diffusers.utils import export_to_video
from diffusers import CogVideoXPipeline, AutoencoderKLCogVideoX
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType


model_id = os.environ.get("COGVIDEOX_DIR", "THUDM/CogVideoX-5b")


def is_cogvideox_1_5():
    return "CogVideoX1.5" in model_id or "THUDM/CogVideoX1.5" in model_id


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


pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B8, good balance between performance and precision
cache_options = CacheType.default_options(CacheType.DBCache)

apply_cache_on_pipe(pipe, **cache_options)

pipe.enable_model_cpu_offload()
assert isinstance(pipe.vae, AutoencoderKLCogVideoX)  # enable type check for IDE
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
    num_frames=(
        # Avoid OOM for CogVideoX1.5 model on 48GB GPU
        16
        if (is_cogvideox_1_5() and get_gpu_memory_in_gib() < 48)
        else 49
    ),
    guidance_scale=6,
    generator=torch.Generator("cuda").manual_seed(0),
).frames[0]

print("Saving video to cogvideox.mp4")
export_to_video(video, "cogvideox.mp4", fps=8)
