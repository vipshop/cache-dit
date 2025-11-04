import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import (
    HunyuanImagePipeline,
    HunyuanImageTransformer2DModel,
)
from diffusers.quantizers import PipelineQuantizationConfig

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

enable_quatization = args.quantize and GiB() < 96
# For now you need to install the latest diffusers as below:
# pip install git+https://github.com/huggingface/diffusers@main
pipe: HunyuanImagePipeline = HunyuanImagePipeline.from_pretrained(
    os.environ.get(
        "HUNYUAN_IMAGE_DIR",
        "hunyuanvideo-community/HunyuanImage-2.1-Diffusers",
    ),
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
        if enable_quatization
        else None
    ),
)

if GiB() < 96:
    if enable_quatization:
        pipe.transformer = cache_dit.quantize(
            pipe.transformer,
            quant_type=args.quantize_type,  # float8_weight_only
        )
        pipe.to(device)
else:
    pipe.to(device)


if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

torch.cuda.empty_cache()
assert isinstance(pipe.transformer, HunyuanImageTransformer2DModel)

if GiB() < 96 and not enable_quatization:
    pipe.enable_model_cpu_offload(device=device)

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(warmup: bool = False):
    prompt = "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, "
    "standing in a painting studio, wearing a red knitted scarf and a red beret with "
    "the word “Tencent” on it, holding a paintbrush with a focused expression as it "
    "paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."
    image = pipe(
        prompt,
        num_inference_steps=50 if not warmup else 5,
        height=2048,
        width=2048,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
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
    save_path = f"hunyuan_image_2.1.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
