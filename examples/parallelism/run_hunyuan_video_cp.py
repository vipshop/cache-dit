import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import (
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
    AutoencoderKLHunyuanVideo,
)
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video
from utils import (
    GiB,
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
)

import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

enable_quatization = args.quantize and GiB() < 96

pipe: HunyuanVideoPipeline = HunyuanVideoPipeline.from_pretrained(
    os.environ.get(
        "HUNYUAN_VIDEO_DIR",
        "hunyuanvideo-community/HunyuanVideo",
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
            components_to_quantize=["text_encoder"],  # 4GiB
        )
        if enable_quatization
        else None
    ),
)


if GiB() < 96:
    if enable_quatization:
        pipe.transformer = cache_dit.quantize(
            pipe.transformer,
            quant_type=args.quantize_type,  # float8_weight_only, 12GiB
        )
        pipe.to(device)
else:
    pipe.to(device)


if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

assert isinstance(pipe.transformer, HunyuanVideoTransformer3DModel)

if GiB() < 96 and not enable_quatization:
    pipe.enable_model_cpu_offload(device=device)

assert isinstance(pipe.vae, AutoencoderKLHunyuanVideo)
pipe.vae.enable_tiling()

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(warmup: bool = False):
    prompt = "A cat walks on the grass, realistic"
    output = pipe(
        prompt,
        height=320,
        width=512,
        num_frames=61,
        num_inference_steps=30 if not warmup else 5,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return output


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

# warmup
_ = run_pipe(warmup=True)

start = time.time()
video = run_pipe()
end = time.time()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"hunyuan_video.{strify(args, pipe)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=15)


maybe_destroy_distributed()
