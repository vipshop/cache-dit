import os
import sys

sys.path.append("..")

import time
import torch
import numpy as np
from diffusers import (
    AutoencoderKLWan,
    ChronoEditTransformer3DModel,
    ChronoEditPipeline,
)
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
from utils import get_args, strify, cachify
import cache_dit


args = get_args()
print(args)


model_id = "nvidia/ChronoEdit-14B-Diffusers"
model_id = os.environ.get("CHRONO_EDIT_DIR", model_id)

image_encoder = CLIPVisionModel.from_pretrained(
    model_id, subfolder="image_encoder", torch_dtype=torch.float32
)
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)
transformer = ChronoEditTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)

pipe = ChronoEditPipeline.from_pretrained(
    model_id,
    vae=vae,
    image_encoder=image_encoder,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

if args.cache:
    cachify(args, pipe)

# Enable memory savings
pipe.enable_model_cpu_offload()
assert isinstance(pipe.vae, AutoencoderKLWan)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = load_image(
    "https://huggingface.co/spaces/nvidia/ChronoEdit/resolve/main/examples/3.png"
)

max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = (
    pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
)
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = (
    "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup's liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
    "The mouse's pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup's floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
)


def run_pipe(warmup: bool = False):
    output = pipe(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=49 if not warmup else 5,
        guidance_scale=5.0,
        enable_temporal_reasoning=False,
        num_temporal_reasoning_steps=0,
        num_inference_steps=50,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return output


# warmup
run_pipe(warmup=True)

start = time.time()
output = run_pipe()
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"chrono-edit.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(output, save_path, fps=16)
