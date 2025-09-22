import os
import sys

sys.path.append("..")

import time
import torch
from diffusers.utils import export_to_video
from diffusers import CogVideoXPipeline, AutoencoderKLCogVideoX
from utils import get_args, strify, cachify
import cache_dit


args = get_args()
print(args)


model_id = os.environ.get("COGVIDEOX_DIR", "THUDM/CogVideoX-2b")

pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

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
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"cogvideox.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=8)
