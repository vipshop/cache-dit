import os
import time
import torch
from diffusers.utils import export_to_video
from diffusers import CogVideoXPipeline, AutoencoderKLCogVideoX
from utils import GiB, get_args
import cache_dit


args = get_args()
print(args)


model_id = os.environ.get("COGVIDEOX_DIR", "THUDM/CogVideoX-5b")


def is_cogvideox_1_5():
    return "CogVideoX1.5" in model_id or "THUDM/CogVideoX1.5" in model_id


pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to("cuda")


if args.cache:
    cache_dit.enable_cache(pipe)
    cache_type_str = "DBCACHE"
else:
    cache_type_str = "NONE"


pipe.enable_model_cpu_offload()
assert isinstance(pipe.vae, AutoencoderKLCogVideoX)  # enable type check for IDE
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

start = time.time()
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
        if (is_cogvideox_1_5() and GiB() <= 48)
        else 49
    ),
    guidance_scale=6,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"cogvideox.{cache_type_str}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=8)
