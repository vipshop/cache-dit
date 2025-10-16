import os
import sys

sys.path.append("..")

import time
import math

import torch
from PIL import Image
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from diffusers import FlowMatchEulerDiscreteScheduler
from nunchaku import NunchakuQwenImageTransformer2DModel

from io import BytesIO
import requests
from utils import get_args, strify
import cache_dit

args = get_args()
print(args)

# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

steps = 8 if args.steps is None else args.steps
assert steps in [8, 4]

nunchaku_qwen_image_edit_plus_dir = os.environ.get(
    "NUNCHAKA_QWEN_IMAGE_EDIT_2509_DIR",
    "nunchaku-tech/nunchaku-qwen-image-edit-2509",
)

transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"{nunchaku_qwen_image_edit_plus_dir}/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-{steps}steps.safetensors"
)

# Minimize VRAM required: 25GiB if use w4a16_text_encoder else 35GiB
w4a16_text_encoder = False
pipe = QwenImageEditPlusPipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_EDIT_2509_DIR",
        "Qwen/Qwen-Image-Edit-2509",
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
            Fn_compute_blocks=16,
            Bn_compute_blocks=16,
            max_warmup_steps=4 if steps > 4 else 2,
            warmup_interval=2 if steps > 4 else 1,
            max_cached_steps=2 if steps > 4 else 1,
            max_continuous_cached_steps=1,
            enable_separate_cfg=False,  # true_cfg_scale=1.0
            residual_diff_threshold=0.50 if steps > 4 else 0.8,
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


def run_pipe():
    inputs = {
        "image": [image1, image2],
        "prompt": prompt,
        "generator": torch.Generator(device="cpu").manual_seed(0),
        "true_cfg_scale": 1.0,
        "negative_prompt": " ",
        "num_inference_steps": steps,
    }
    return pipe(**inputs).images[0]


if args.compile:
    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
    cache_dit.set_compile_configs()
    pipe.transformer.compile_repeated_blocks(mode="default")

    # Warmup
    run_pipe()

start = time.time()
image = run_pipe()
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image-edit-plus-lightning.{steps}steps.nunchaku.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
