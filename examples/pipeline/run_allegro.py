import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import AllegroPipeline
from diffusers.utils import export_to_video
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)

model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get("ALLEGRO_DIR", "rhymes-ai/Allegro")
)

pipe = AllegroPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)

pipe.to("cuda")

pipe.vae.enable_tiling()

if args.cache:
    cachify(args, pipe)

prompt = (
    "A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, "
    "the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this "
    "location might be a popular spot for docking fishing boats."
)

if args.prompt is not None:
    prompt = args.prompt

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
video = pipe(
    prompt,
    guidance_scale=7.5,
    max_sequence_length=512,
    num_inference_steps=100,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"allegro.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=8)
