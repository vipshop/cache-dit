import os
import sys

sys.path.append("..")

import time
import torch
import torch.distributed as dist
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel,
)
from utils import (
    get_args,
    strify,
    maybe_init_distributed,
    maybe_destroy_distributed,
)
import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

nunchaku_qwen_image_dir = os.environ.get(
    "NUNCHAKA_QWEN_IMAGE_DIR",
    "nunchaku-tech/nunchaku-qwen-image",
)
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"{nunchaku_qwen_image_dir}/svdq-int4_r32-qwen-image.safetensors"
)

# Minimize VRAM required: 20GiB if use w4a16_text_encoder else 30GiB
w4a16_text_encoder = False
pipe = QwenImagePipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_DIR",
        "Qwen/Qwen-Image",
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


if args.cache or args.parallel_type is not None:
    from cache_dit import (
        DBCacheConfig,
        ParallelismConfig,
        TaylorSeerCalibratorConfig,
    )

    cache_dit.enable_cache(
        pipe,
        cache_config=(
            DBCacheConfig(
                Fn_compute_blocks=args.Fn,
                Bn_compute_blocks=args.Bn,
                max_warmup_steps=args.max_warmup_steps,
                max_cached_steps=args.max_cached_steps,
                max_continuous_cached_steps=args.max_continuous_cached_steps,
                residual_diff_threshold=args.rdt,
            )
            if args.cache
            else None
        ),
        calibrator_config=(
            TaylorSeerCalibratorConfig(
                taylorseer_order=args.taylorseer_order,
            )
            if args.taylorseer
            else None
        ),
        parallelism_config=(
            ParallelismConfig(
                ulysses_size=(
                    dist.get_world_size()
                    if args.parallel_type == "ulysses"
                    else None
                ),
                ring_size=(
                    dist.get_world_size()
                    if args.parallel_type == "ring"
                    else None
                ),
            )
            if args.parallel_type in ["ulysses", "ring"]
            else None
        ),
    )


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
        num_inference_steps=(
            (50 if args.steps is None else args.steps) if not warmup else 5
        ),
        true_cfg_scale=4.0,
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
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"qwen-image.nunchaku.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    if not args.perf:
        print(f"Saving image to {save_path}")
        image.save(save_path)

maybe_destroy_distributed()
