import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
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

model_id = "genmo/mochi-1-preview"
model_id = os.environ.get("MOCHI_DIR", model_id)

pipe = MochiPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to(device)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

pipe.enable_vae_tiling()

torch.cuda.empty_cache()

prompt = (
    "Close-up of a chameleon's eye, with its scaly skin "
    "changing color. Ultra high resolution 4k."
)


def run_pipe():
    video = pipe(
        prompt,
        num_frames=49,
        num_inference_steps=64,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return video


start = time.time()
video = run_pipe()
end = time.time()

if rank == 0:
    stats = cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"mochi.{strify(args, stats)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=10)

maybe_destroy_distributed()
