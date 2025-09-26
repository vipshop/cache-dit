import os
import sys

sys.path.append("..")

import time

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from io import BytesIO
import requests
from utils import GiB, get_args, strify, cachify
import cache_dit

args = get_args()
print(args)

pipe = QwenImageEditPlusPipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_EDIT_2509_DIR",
        "Qwen/Qwen-Image-Edit-2509",
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
)

if args.cache:
    cachify(args, pipe)

if torch.cuda.device_count() <= 1:
    # Enable memory savings
    pipe.enable_model_cpu_offload()

image1 = Image.open(
    BytesIO(
        requests.get(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_1.jpg"
        ).content
    )
)
image2 = Image.open(
    BytesIO(
        requests.get(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/edit2509/edit2509_2.jpg"
        ).content
    )
)
prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."
inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.Generator(device="cpu").manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

if args.compile:
    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 8192
    pipe.transformer.compile_repeated_blocks(mode="default")

    # Warmup
    image = pipe(**inputs).images[0]

start = time.time()
image = pipe(**inputs).images[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image-edit-plus.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
