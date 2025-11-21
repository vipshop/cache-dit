import os
import sys

sys.path.append("..")

import time

import torch
from PIL import Image
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from nunchaku import NunchakuQwenImageTransformer2DModel

from io import BytesIO
import requests
from utils import get_args, strify, MemoryTracker
import cache_dit

args = get_args()
print(args)


nunchaku_qwen_image_edit_plus_dir = os.environ.get(
    "NUNCHAKA_QWEN_IMAGE_EDIT_2509_DIR",
    "nunchaku-tech/nunchaku-qwen-image-edit-2509",
)

transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"{nunchaku_qwen_image_edit_plus_dir}/svdq-int4_r128-qwen-image-edit-2509.safetensors"
)

# Minimize VRAM required: 20GiB if use w4a16_text_encoder else 30GiB
w4a16_text_encoder = False
pipe = QwenImageEditPlusPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "QWEN_IMAGE_EDIT_2509_DIR",
            "Qwen/Qwen-Image-Edit-2509",
        )
    ),
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
        if w4a16_text_encoder
        else None
    ),
).to("cuda")

if args.cache:
    from cache_dit import (
        DBCacheConfig,
        TaylorSeerCalibratorConfig,
    )

    cache_dit.enable_cache(
        pipe,
        cache_config=DBCacheConfig(
            Fn_compute_blocks=args.Fn,
            Bn_compute_blocks=args.Bn,
            max_warmup_steps=args.max_warmup_steps,
            max_cached_steps=args.max_cached_steps,
            max_continuous_cached_steps=args.max_continuous_cached_steps,
            residual_diff_threshold=args.rdt,
        ),
        calibrator_config=(
            TaylorSeerCalibratorConfig(
                taylorseer_order=args.taylorseer_order,
            )
            if args.taylorseer
            else None
        ),
    )


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


def run_pipe():
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
    return pipe(**inputs).images[0]


if args.compile:
    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
    cache_dit.set_compile_configs()
    pipe.transformer.compile_repeated_blocks(mode="default")

    # Warmup
    run_pipe()

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image-edit-plus.nunchaku.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
