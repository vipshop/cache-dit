import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Kandinsky5T2VPipeline, AutoencoderKLHunyuanVideo
from diffusers.utils import export_to_video
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit

args = get_args()
print(args)

# Available models:
# ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers
# ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers
# ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers
# ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s-Diffusers

model_id = (
    args.model_path
    if args.model_path is not None
    else "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
)
model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get("KANDINSKY5_T2V_DIR", model_id)
)
pipe = Kandinsky5T2VPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe = pipe.to("cuda")

if args.cache:
    cachify(args, pipe, enable_separate_cfg=not ("nocfg" in model_id))

prompt = "A cat and a dog baking a cake together in a kitchen."

if args.prompt is not None:

    prompt = args.prompt
negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"
if args.negative_prompt is not None:
    negative_prompt = args.negative_prompt

assert isinstance(pipe.vae, AutoencoderKLHunyuanVideo)

pipe.vae.enable_tiling()


def run_pipe():
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=768,
        num_frames=121,
        num_inference_steps=50,
        guidance_scale=5.0,
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

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"kandinsky5.{strify(args, pipe)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=24, quality=9)
