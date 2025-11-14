import os
import time
import torch
import numpy as np
from PIL import Image
import torch.distributed as dist
from diffusers import (
    AutoencoderKLWan,
    ChronoEditTransformer3DModel,
    ChronoEditPipeline,
)
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import ContextParallelConfig
from diffusers.utils import load_image
from transformers import CLIPVisionModel


dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = torch.device("cuda", rank % torch.cuda.device_count())
world_size = dist.get_world_size()
torch.cuda.set_device(device)

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
            # text_encoder: ~ 6GiB, transformer: ~ 8GiB, total: ~14GiB
            components_to_quantize=["text_encoder", "transformer"],
        )
    ),
).to(device)

torch.cuda.empty_cache()
assert isinstance(pipe.vae, AutoencoderKLWan)
pipe.vae.enable_tiling()

image = load_image("../examples/data/chrono_edit_example.png")

max_area = 720 * 1280
aspect_ratio = image.height / image.width
mod_value = (
    pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
)
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = (
    "The user wants to transform the image by adding a small, cute mouse sitting inside the floral teacup, enjoying a spa bath. The mouse should appear relaxed and cheerful, with a tiny white bath towel draped over its head like a turban. It should be positioned comfortably in the cup’s liquid, with gentle steam rising around it to blend with the cozy atmosphere. "
    "The mouse’s pose should be natural—perhaps sitting upright with paws resting lightly on the rim or submerged in the tea. The teacup’s floral design, gold trim, and warm lighting must remain unchanged to preserve the original aesthetic. The steam should softly swirl around the mouse, enhancing the spa-like, whimsical mood."
)

assert isinstance(pipe.transformer, ChronoEditTransformer3DModel)
pipe.transformer.set_attention_backend("native")
if world_size > 1:
    pipe.transformer.enable_parallelism(
        config=ContextParallelConfig(ulysses_degree=world_size)
    )

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
        num_inference_steps=50 if not warmup else 2,
        generator=torch.Generator("cuda").manual_seed(0),
    ).frames[0]
    output = Image.fromarray((output[-1] * 255).clip(0, 255).astype("uint8"))
    return output


start = time.time()
output = run_pipe()
end = time.time()

if rank == 0:
    time_cost = end - start
    save_path = f"chrono-edit.{world_size}gpus.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    output.save(save_path)

if dist.is_initialized():
    dist.destroy_process_group()
