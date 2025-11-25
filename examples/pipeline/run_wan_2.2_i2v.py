import os
import sys

sys.path.append("..")

import time
import torch
import diffusers
from diffusers import (
    AutoencoderKLWan,
    WanTransformer3DModel,
    WanImageToVideoPipeline,
)
from diffusers.utils import export_to_video, load_image

from utils import get_args, GiB, strify, cachify, MemoryTracker
import cache_dit
import numpy as np

args = get_args()
print(args)

model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get(
        "WAN_2_2_I2V_DIR",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    )
)

pipe: WanImageToVideoPipeline = WanImageToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # Based on: https://github.com/huggingface/diffusers/pull/12523
    device_map=("balanced" if GiB() < 96 and torch.cuda.device_count() > 1 else None),
)

# When device_map is None, we need to explicitly move the model to GPU
# or enable CPU offload to avoid running on CPU
if GiB() < 96 and torch.cuda.device_count() <= 1:
    # issue: https://github.com/huggingface/diffusers/issues/12499
    print("Enable model cpu offload for low memory device.")
    pipe.enable_model_cpu_offload()
elif torch.cuda.device_count() > 1 and pipe.device.type == "cpu":
    # Multi-GPU but model is on CPU (device_map was None): move to default GPU
    pipe.to("cuda")


if args.cache:
    from cache_dit import (
        ForwardPattern,
        BlockAdapter,
        ParamsModifier,
        DBCacheConfig,
    )

    cachify(
        args,
        BlockAdapter(
            pipe=pipe,
            transformer=[
                pipe.transformer,
                pipe.transformer_2,
            ],
            blocks=[
                pipe.transformer.blocks,
                pipe.transformer_2.blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_2,
                ForwardPattern.Pattern_2,
            ],
            params_modifiers=[
                # high-noise transformer only have 30% steps
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=4,
                        max_cached_steps=8,
                    ),
                ),
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=2,
                        max_cached_steps=20,
                    ),
                ),
            ],
            has_separate_cfg=True,
        ),
    )

# Wan currently requires installing diffusers from source
assert isinstance(pipe.vae, AutoencoderKLWan)  # enable type check for IDE
if diffusers.__version__ >= "0.34.0":
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
else:
    print(
        "Wan pipeline requires diffusers version >= 0.34.0 "
        "for vae tiling and slicing, please install diffusers "
        "from source."
    )

assert isinstance(pipe.transformer, WanTransformer3DModel)
assert isinstance(pipe.transformer_2, WanTransformer3DModel)

if args.quantize:
    assert isinstance(args.quantize_type, str)
    if args.quantize_type.endswith("wo"):  # weight only
        pipe.transformer = cache_dit.quantize(
            pipe.transformer,
            quant_type=args.quantize_type,
        )
    # We only apply activation quantization (default: FP8 DQ)
    # for low-noise transformer to avoid non-trivial precision
    # downgrade.
    pipe.transformer_2 = cache_dit.quantize(
        pipe.transformer_2,
        quant_type=args.quantize_type,
    )


image = load_image(
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
)

max_area = 480 * 832
aspect_ratio = image.height / image.width
mod_value = pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
image = image.resize((width, height))

prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
if args.prompt is not None:
    prompt = args.prompt
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
if args.negative_prompt is not None:
    negative_prompt = args.negative_prompt


def run_pipe():
    video = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=81,  # pipe.vae_scale_factor_temporal=4
        guidance_scale=3.5,
        num_inference_steps=50,
        generator=torch.Generator(device="cpu").manual_seed(0),
    ).frames[0]

    return video


if args.compile or args.quantize:
    cache_dit.set_compile_configs()
    pipe.transformer.compile_repeated_blocks(fullgraph=True)
    pipe.transformer_2.compile_repeated_blocks(fullgraph=True)

    # warmup
    run_pipe()

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
video = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe, details=True)

time_cost = end - start
save_path = f"wan2.2-i2v.frame{len(video)}.{height}x{width}.{strify(args, pipe)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=16)
