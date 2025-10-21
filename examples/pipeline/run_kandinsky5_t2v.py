import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Kandinsky5T2VPipeline
from diffusers.utils import export_to_video
from utils import get_args, strify, cachify
import cache_dit

args = get_args()
print(args)

# Available models:
# ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers
# ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers
# ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers
# ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s-Diffusers

model_id = "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
model_id = os.environ.get("KANDINSKY5_T2V_DIR", model_id)
pipe = Kandinsky5T2VPipeline.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

if args.cache:
    cachify(
        args,
        pipe,
        enable_separate_cfg=(True if "nocfg" not in model_id else False),
    )

prompt = "A cat and a dog baking a cake together in a kitchen."
negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"


def run_pipe():
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=768,
        num_frames=121,
        num_inference_steps=50,
        guidance_scale=5.0,
    ).frames[0]
    return video


start = time.time()
video = run_pipe()
end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"kandinsky5.{strify(args, pipe)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=24, quality=9)
