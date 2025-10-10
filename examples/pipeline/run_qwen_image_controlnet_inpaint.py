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
)

assert isinstance(pipe.controlnet, QwenImageControlNetModel)
assert isinstance(pipe.transformer, QwenImageTransformer2DModel)

control_image = load_image(
    "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/images/image1.png"
)
mask_image = load_image(
    "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets/masks/mask1.png"
)
prompt = "一辆绿色的出租车行驶在路上"
negative_prompt = (
    "worst quality, low quality, blurry, text, watermark, logo"  # or " "
)


if GiB() < 96:
    # FP8 weight only
    if args.quantize:
        # Minimum VRAM required: 42 GiB
        args.quantize_type = "fp8_w8a16_wo"  # force
        pipe.transformer = cache_dit.quantize(
            pipe.transformer,
            quant_type="fp8_w8a16_wo",
            exclude_layers=[
                "img_in",
                "txt_in",
                "embedder",
                "embed",
                "norm_out",
                "proj_out",
            ],
        )
        pipe.text_encoder = cache_dit.quantize(
            pipe.text_encoder,
            quant_type="int4_w4a16_wo",  # fp8_w8a16_wo
        )

        pipe.to("cuda")
    else:
        print("Enable Model CPU Offload ...")
        pipe.enable_model_cpu_offload()
    pipe.enable_vae_tiling()
else:
    pipe.to("cuda")

if args.cache:

    cachify(
        args,
        pipe,
        # do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        # (negative_prompt is not None, default None)
        enable_separate_cfg=False if negative_prompt is None else True,
    )


start = time.time()
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    control_image=control_image.convert("RGB"),
    control_mask=mask_image,
    controlnet_conditioning_scale=1.0,
    width=mask_image.size[0],
    height=mask_image.size[1],
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(0),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image-controlnet-inpaint.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
