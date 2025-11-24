import os
import sys

sys.path.append("..")

import time

import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from io import BytesIO
import requests
from utils import GiB, get_args, strify, cachify, MemoryTracker
import cache_dit

args = get_args()
print(args)

pipe = QwenImageEditPlusPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "QWEN_IMAGE_EDIT_2509_DIR",
            "Qwen/Qwen-Image-Edit-2509",
        )
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=("balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None),
)

if args.cache:
    cachify(args, pipe)

# When device_map is None, we need to explicitly move the model to GPU
# or enable CPU offload to avoid running on CPU
if torch.cuda.device_count() <= 1:
    # Single GPU: use CPU offload for memory efficiency
    pipe.enable_model_cpu_offload()
elif torch.cuda.device_count() > 1 and pipe.device.type == "cpu":
    # Multi-GPU but model is on CPU (device_map was None): move to default GPU
    pipe.to("cuda")

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
if args.prompt is not None:
    prompt = args.prompt
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

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(**inputs).images[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image-edit-plus.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
