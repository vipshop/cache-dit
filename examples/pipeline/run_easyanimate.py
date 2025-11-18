import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import EasyAnimatePipeline
from diffusers.utils import export_to_video
from utils import get_args, strify, cachify
import cache_dit


args = get_args()
print(args)

model_id = os.environ.get("EASY_ANIMATE_DIR", "alibaba-pai/EasyAnimateV5.1-7b-zh")


pipe = EasyAnimatePipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)

pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

prompt = "A cat walks on the grass, realistic style."
negative_prompt = "bad detailed"

start = time.time()
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=49,
    num_inference_steps=30,
    generator=torch.Generator("cuda").manual_seed(0),
).frames[0]

end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"easyanimate.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=8)
