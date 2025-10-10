import os
import sys

sys.path.append("..")

import time
import torch
from diffusers.utils import load_image
from diffusers import (
    QwenImageControlNetModel,
    QwenImageControlNetInpaintPipeline,
    QwenImageTransformer2DModel,
)
from diffusers.quantizers import PipelineQuantizationConfig

from utils import GiB, get_args, strify, cachify

import cache_dit


args = get_args()
print(args)

base_model = os.environ.get(
    "QWEN_IMAGE_DIR",
    "Qwen/Qwen-Image",
)
controlnet_model = os.environ.get(
    "QWEN_IMAGE_CN_DIR",
    "InstantX/Qwen-Image-ControlNet-Inpainting",
)

controlnet = QwenImageControlNetModel.from_pretrained(
    controlnet_model,
    torch_dtype=torch.bfloat16,
)

pipe = QwenImageControlNetInpaintPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.bfloat16,
    quantization_config=(
        PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=[
                "transformer",
                "controlnet",
                "text_encoder",
            ],
        )
        if GiB() < 96
        else None
    ),
)

pipe.to("cuda")

assert isinstance(pipe.controlnet, QwenImageControlNetModel)
assert isinstance(pipe.transformer, QwenImageTransformer2DModel)

if args.cache:
    cachify(args, pipe)


control_image = load_image(
    "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/images/image1.png"
)
mask_image = load_image(
    "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/masks/mask1.png"
)
prompt = "一辆绿色的出租车行驶在路上"

start = time.time()
# do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
image = pipe(
    prompt=prompt,
    negative_prompt=" ",
    control_image=control_image,
    control_mask=mask_image,
    controlnet_conditioning_scale=1.0,
    width=control_image.size[0],
    height=control_image.size[1],
    num_inference_steps=30,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image-controlnet-inpaint.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
