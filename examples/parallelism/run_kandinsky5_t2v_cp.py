import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import (
    Kandinsky5T2VPipeline,
    AutoencoderKLHunyuanVideo,
    Kandinsky5Transformer3DModel,
)
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video
from utils import (
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

# Available models:
# ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers
# ai-forever/Kandinsky-5.0-T2V-Lite-nocfg-5s-Diffusers
# ai-forever/Kandinsky-5.0-T2V-Lite-distilled16steps-5s-Diffusers
# ai-forever/Kandinsky-5.0-T2V-Lite-pretrain-5s-Diffusers

model_id = "ai-forever/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers"
model_id = os.environ.get("KANDINSKY5_T2V_DIR", model_id)
pipe = Kandinsky5T2VPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["text_encoder", "text_encoder_2"],
    ),
)
pipe = pipe.to("cuda")

if args.cache or args.parallel_type is not None:
    cachify(args, pipe, enable_separate_cfg=not ("nocfg" in model_id))

prompt = "A cat and a dog baking a cake together in a kitchen."
negative_prompt = "Static, 2D cartoon, cartoon, 2d animation, paintings, images, worst quality, low quality, ugly, deformed, walking backwards"

assert isinstance(pipe.vae, AutoencoderKLHunyuanVideo)

pipe.vae.enable_tiling()


def run_pipe(warmup: bool = False):
    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=512,
        width=768,
        num_frames=121,
        num_inference_steps=50 if not warmup else 5,
        guidance_scale=5.0,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return video


if args.compile:
    cache_dit.set_compile_configs()
    assert isinstance(pipe.transformer, Kandinsky5Transformer3DModel)
    pipe.transformer.compile_repeated_blocks(mode="default")


# warmup
_ = run_pipe(warmup=True)

start = time.time()
video = run_pipe()
end = time.time()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"kandinsky5.{strify(args, pipe)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=24, quality=9)

maybe_destroy_distributed()
