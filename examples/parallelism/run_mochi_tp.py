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
    MemoryTracker,
)

import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

model_id = args.model_path if args.model_path is not None else "genmo/mochi-1-preview"
model_id = args.model_path if args.model_path is not None else os.environ.get("MOCHI_DIR", model_id)

pipe = MochiPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

pipe.enable_model_cpu_offload(device=device)
pipe.vae.enable_tiling()

torch.cuda.empty_cache()

prompt = (
    "Close-up of a chameleon's eye, with its scaly skin "
    "changing color. Ultra high resolution 4k."
)


if args.prompt is not None:

    prompt = args.prompt


def run_pipe():
    video = pipe(
        prompt,
        num_frames=49,
        num_inference_steps=64,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return video


memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
video = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

if rank == 0:
    stats = cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"mochi.{strify(args, stats)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=10)

maybe_destroy_distributed()
