import os
import sys

sys.path.append("..")

import time
import torch
import numpy as np
from PIL import Image
from diffusers import (
    AutoencoderKLWan,
    ChronoEditTransformer3DModel,
    ChronoEditPipeline,
)
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import load_image
from transformers import CLIPVisionModel
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
    quantization_config=(
        PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["text_encoder"],
        )
        if args.quantize
        else None
    ),
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

# Enable memory savings
pipe.enable_model_cpu_offload(device=device)
assert isinstance(pipe.vae, AutoencoderKLWan)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

image = load_image("../data/chrono_edit_example.png")

max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = (
    pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
)
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = "Transform to high-end PVC scale figure."

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(warmup: bool = False):
    output = pipe(
        image=image,
        prompt=prompt,
        height=height,
        width=width,
        num_frames=5,
        guidance_scale=5.0,
        enable_temporal_reasoning=False,
        num_temporal_reasoning_steps=0,
        num_inference_steps=50 if not warmup else 5,
        generator=torch.Generator("cpu").manual_seed(42),
    ).frames[0]
    output = Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8"))
    return output


# warmup
run_pipe(warmup=True)

start = time.time()
output = run_pipe()
end = time.time()

if rank == 0:
    stats = cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"chrono-edit.{strify(args, stats)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    output.save(save_path)

maybe_destroy_distributed()
