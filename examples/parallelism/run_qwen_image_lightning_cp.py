import os
import sys

sys.path.append("..")

import time
import torch
import math
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    AutoencoderKLQwenImage,
    FlowMatchEulerDiscreteScheduler,
)
from utils import (
    GiB,
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
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

pipe = QwenImagePipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_DIR",
        "Qwen/Qwen-Image",
    ),
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
)

assert isinstance(pipe.transformer, QwenImageTransformer2DModel)

steps = 8 if args.steps is None else args.steps
assert steps in [8, 4]

pipe.load_lora_weights(
    os.environ.get(
        "QWEN_IMAGE_LIGHT_DIR",
        "lightx2v/Qwen-Image-Lightning",
    ),
    weight_name=(
        "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors"
        if steps > 4
        else "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
    ),
)

pipe.fuse_lora()
pipe.unload_lora_weights()

enable_quatization = args.quantize and GiB() < 96
if GiB() < 96:
    if enable_quatization:
        pipe.transformer = cache_dit.quantize(
            pipe.transformer,
            quant_type=args.quantize_type,
            exclude_layers=[
                "img_in",
                "txt_in",
            ],
        )
        pipe.text_encoder = cache_dit.quantize(
            pipe.text_encoder,
            quant_type=args.quantize_type,
        )
        pipe.to(device)
else:
    pipe.to(device)

if GiB() <= 48 and not enable_quatization:
    assert isinstance(pipe.vae, AutoencoderKLQwenImage)
    pipe.vae.enable_tiling()

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


if GiB() < 96 and not enable_quatization:
    # NOTE: Enable cpu offload before enabling context parallelism will
    # raise shape error after first pipe call, so we enable it after.
    # It seems a bug of diffusers that cpu offload is not fully
    # compatible with context parallelism, visa versa.
    pipe.enable_model_cpu_offload(device=device)


positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
}

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

# using an empty string if you do not have specific concept to remove
negative_prompt = " "

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(warmup: bool = False):
    # do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    output = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=1024 if args.width is None else args.width,
        height=1024 if args.height is None else args.height,
        num_inference_steps=steps if not warmup else steps,
        true_cfg_scale=1.0,  # means no separate cfg
        generator=torch.Generator(device="cpu").manual_seed(0),
        output_type="latent" if args.perf else "pil",
    )
    image = output.images[0] if not args.perf else None
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)


# warmup
_ = run_pipe(warmup=True)

start = time.time()
image = run_pipe()
end = time.time()


if rank == 0:
    stats = cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"qwen-image-lightning.{steps}steps.{strify(args, stats)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
