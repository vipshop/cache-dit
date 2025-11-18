import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Cosmos2VideoToWorldPipeline
from diffusers.utils import export_to_video, load_image
from utils import get_args, strify, cachify
import cache_dit


args = get_args()
print(args)

# Available checkpoints: nvidia/Cosmos-Predict2-2B-Video2World, nvidia/Cosmos-Predict2-14B-Video2World
model_id = os.environ.get("COSMOS_DIR", "nvidia/Cosmos-Predict2-2B-Video2World")

pipe = Cosmos2VideoToWorldPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

prompt = "A close-up shot captures a vibrant yellow scrubber vigorously working on a grimy plate, its bristles moving in circular motions to lift stubborn grease and food residue. The dish, once covered in remnants of a hearty meal, gradually reveals its original glossy surface. Suds form and bubble around the scrubber, creating a satisfying visual of cleanliness in progress. The sound of scrubbing fills the air, accompanied by the gentle clinking of the dish against the sink. As the scrubber continues its task, the dish transforms, gleaming under the bright kitchen lights, symbolizing the triumph of cleanliness over mess."
negative_prompt = "The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality."
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/yellow-scrubber.png"
)

start = time.time()
video = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.Generator().manual_seed(1),
).frames[0]

end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"cosmos.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=8)
