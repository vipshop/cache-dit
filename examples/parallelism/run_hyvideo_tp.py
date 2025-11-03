import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
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

pipe: HunyuanVideoPipeline = HunyuanVideoPipeline.from_pretrained(
    os.environ.get(
        "HUNYUAN_VIDEO_DIR",
        "hunyuanvideo-community/HunyuanVideo",
    ),
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

assert isinstance(pipe.transformer, HunyuanVideoTransformer3DModel)
pipe.enable_model_cpu_offload(device=device)
pipe.vae.enable_tiling()

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(pipe: HunyuanVideoPipeline):
    prompt = "A cat walks on the grass, realistic"
    output = pipe(
        prompt,
        height=320,
        width=512,
        num_frames=61,
        num_inference_steps=30,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return output


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

# warmup
_ = run_pipe(pipe)

start = time.time()
video = run_pipe(pipe)
end = time.time()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"hunyuan_video.{strify(args, pipe)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=15)


maybe_destroy_distributed()
