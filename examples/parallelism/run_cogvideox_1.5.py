import os
import sys

sys.path.append("..")

import time
import torch
from diffusers.utils import export_to_video
from diffusers import CogVideoXPipeline, AutoencoderKLCogVideoX
from utils import (
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
)
import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

pipe = CogVideoXPipeline.from_pretrained(
    os.environ.get("COGVIDEOX_1_5_DIR", "zai-org/CogVideoX1.5-5B"),
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

assert isinstance(pipe.vae, AutoencoderKLCogVideoX)  # enable type check for IDE
pipe.enable_model_cpu_offload(device=device)
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


def run_pipe(warmup: bool = False):
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=50 if not warmup else 5,
        num_frames=16,
        guidance_scale=6,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return video


# warmup
_ = run_pipe(warmup=True)


start = time.time()
video = run_pipe()
end = time.time()

if rank == 0:
    stats = cache_dit.summary(pipe)

    time_cost = end - start
    parallel_type = args.parallel_type or "none"
    save_path = f"cogvideox_1.5_{parallel_type}.{strify(args, stats)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=8)

maybe_destroy_distributed()
