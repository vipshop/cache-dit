import os
import sys

sys.path.append("..")

import time
import torch
import math
from PIL import Image
from io import BytesIO
import requests
from diffusers import (
    QwenImageEditPlusPipeline,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)

from utils import (
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
    pipe_quant_bnb_4bit_config,
    is_optimzation_flags_enabled,
    MemoryTracker,
)
import cache_dit


args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

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


model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get("QWEN_IMAGE_EDIT_2509_DIR", "Qwen/Qwen-Image-Edit-2509")
)

pipe = QwenImageEditPlusPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
    quantization_config=pipe_quant_bnb_4bit_config(args),
)

assert isinstance(pipe.transformer, QwenImageTransformer2DModel)


steps = 8 if args.steps is None else args.steps
assert steps in [8, 4]

pipe.load_lora_weights(
    os.path.join(
        os.environ.get("QWEN_IMAGE_LIGHT_DIR", "lightx2v/Qwen-Image-Lightning"),
        "Qwen-Image-Edit-2509",
    ),
    weight_name=(
        "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
        if steps > 4
        else "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
    ),
)

pipe.fuse_lora()
pipe.unload_lora_weights()

# Apply cache and parallelism here
if is_optimzation_flags_enabled(args):
    from cache_dit import DBCacheConfig

    cachify(
        args,
        pipe,
        cache_config=(
            DBCacheConfig(
                Fn_compute_blocks=16,
                Bn_compute_blocks=16,
                max_warmup_steps=4 if steps > 4 else 2,
                max_cached_steps=2 if steps > 4 else 1,
                max_continuous_cached_steps=1,
                enable_separate_cfg=False,  # true_cfg_scale=1.0
                residual_diff_threshold=0.50 if steps > 4 else 0.8,
            )
            if args.cache
            else None
        ),
    )


width = 1024 if args.width is None else args.width
height = 1024 if args.height is None else args.height

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


pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe():
    inputs = {
        "image": [image1, image2],
        "prompt": prompt,
        "generator": torch.Generator(device="cpu").manual_seed(0),
        "true_cfg_scale": 1.0,  # means no separate cfg for lightning models
        "negative_prompt": " ",
        "num_inference_steps": steps,
        "height": height,
        "width": width,
    }
    output = pipe(**inputs)
    image = output.images[0] if not args.perf else None
    return image


# warmup
_ = run_pipe()

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()


if rank == 0:
    stats = cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"qwen-image-edit-lightning.{steps}steps.{strify(args, stats)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
