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
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen2_5_VLForConditionalGeneration
from diffusers.quantizers import PipelineQuantizationConfig

from utils import (
    GiB,
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
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

quantization_config = (
    (
        PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            # Always use bnb 4bit quantization for text encoder when quantizing to
            # better compatibility for devices like NVIDIA L20 that VRAM <= 48GB.
            components_to_quantize=(
                ["text_encoder", "transformer"]
                if args.quantize_type == "bitsandbytes_4bit"
                else ["text_encoder"]
            ),
        )
    )
    if args.quantize
    else None
)

pipe = QwenImageEditPlusPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
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

assert args.fuse_lora, "Fuse lora must be enabled for tensor parallelism."

if args.fuse_lora:
    pipe.fuse_lora()
    pipe.unload_lora_weights()


# Apply cache and context parallelism here
if args.cache or args.parallel_type is not None:
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


# WARN: Must apply quantization after tensor parallelism is applied.
# torchao is compatible with tensor parallelism but requires to be
# applied after TP.
if args.quantize and args.quantize_type != "bitsandbytes_4bit":
    # Quantize the transformer according to custom quantize
    # type passed from args.
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
        quant_type=args.quantize_type,
        exclude_layers=[
            "img_in",
            "txt_in",
        ],
    )

if GiB() < 48 and not (args.quantize or args.parallel_text_encoder):
    # NOTE: Enable cpu offload before enabling tensor parallelism will
    # raise shape error after first pipe call, so we enable it after.
    # It seems a bug of diffusers that cpu offload is not fully
    # compatible with context parallelism, visa versa.
    assert (
        not args.compile
    ), "Cannot enable compile with cpu offload due to the compatibility issue."
    pipe.enable_model_cpu_offload(device=device)
    print("Enabled model CPU offload.")
else:
    pipe.to(device)


width = 1024 if args.width is None else args.width
height = 1024 if args.height is None else args.height


if GiB() <= 48 and not args.quantize:
    assert isinstance(pipe.vae, AutoencoderKLQwenImage)
    pipe.vae.enable_tiling()

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


if args.compile:
    cache_dit.set_compile_configs()
    torch.set_float32_matmul_precision("high")
    pipe.transformer = torch.compile(pipe.transformer)
    if args.compile_vae:
        pipe.vae.encoder = torch.compile(pipe.vae.encoder)
        pipe.vae.decoder = torch.compile(pipe.vae.decoder)
    if args.compile_text_encoder:
        assert isinstance(pipe.text_encoder, Qwen2_5_VLForConditionalGeneration)
        # NOTE: .tolist() op in visual model will raise spamming warnings, so we temporarily
        # disable compiling visual model here.
        # pipe.text_encoder.model.visual = torch.compile(pipe.text_encoder.model.visual)
        pipe.text_encoder.model.visual = torch.compile(pipe.text_encoder.model.visual)
        pipe.text_encoder.model.language_model = torch.compile(
            pipe.text_encoder.model.language_model
        )

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
