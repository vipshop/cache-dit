import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import CogVideoXPipeline
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
    os.environ.get("COGVIDEOX_DIR", "zai-org/CogVideoX1.5-5B"),
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe, enable_separate_cfg=True)

# Handle model placement based on parallelism type
torch.cuda.empty_cache()
pipe.enable_model_cpu_offload(device=device)

prompt = "A vibrant cherry red sports car sits proudly under the gleaming sun, its polished exterior smooth and flawless, casting a mirror-like reflection. The car features a low, aerodynamic body, angular headlights that gaze forward like predatory eyes, and a set of black, high-gloss racing rims that contrast starkly with the red. A subtle hint of chrome embellishes the grille and exhaust, while the tinted windows suggest a luxurious and private interior. The scene conveys a sense of speed and elegance, the car appearing as if it's about to burst into a sprint along a coastal road, with the ocean's azure waves crashing in the background."


def run_pipe(warmup: bool = False):
    video = pipe(
        prompt=prompt,
        guidance_scale=6.0,  # CogVideoX typically uses higher guidance
        num_inference_steps=50 if not warmup else 5,
        num_videos_per_prompt=1,
        num_frames=49,  # CogVideoX standard frame count
        height=480,  # CogVideoX standard resolution
        width=720,  # CogVideoX standard resolution
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return video


# warmup
run_pipe(warmup=True)

start = time.time()
video = run_pipe()
end = time.time()

if rank == 0:
    stats = cache_dit.summary(pipe)

    time_cost = end - start
    parallel_type = args.parallel_type or "none"
    save_path = f"cogvideox_1.5_{parallel_type}.{strify(args, stats)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving to {save_path}")

    # Save video frames
    export_to_video = pipe.export_to_video
    export_to_video(video, save_path, fps=8)

maybe_destroy_distributed()
