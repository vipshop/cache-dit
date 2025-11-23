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
    MemoryTracker,
    print_rank0,
)
import cache_dit

args = get_args()
rank, device = maybe_init_distributed(args)
print_rank0(args)

pipe = CogVideoXPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get("COGVIDEOX_1_5_DIR", "zai-org/CogVideoX1.5-5B")
    ),
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

assert isinstance(pipe.vae, AutoencoderKLCogVideoX)  # enable type check for IDE
torch.cuda.empty_cache()
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


if args.prompt is not None:

    prompt = args.prompt


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


memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
video = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report(rank)

if rank == 0:
    stats = cache_dit.summary(pipe)

    time_cost = end - start
    parallel_type = args.parallel_type or "none"
    save_path = f"cogvideox_1.5_{parallel_type}.{strify(args, stats)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=8)

maybe_destroy_distributed()
