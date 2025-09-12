import os
import sys

sys.path.append("..")

import time
import torch
import diffusers
import argparse
import numpy as np
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel

from utils import get_args, strify
import cache_dit


def aspect_ratio_resize(image, pipe, max_area=720 * 1280):
    aspect_ratio = image.height / image.width
    mod_value = (
        pipe.vae_scale_factor_spatial * pipe.transformer.config.patch_size[1]
    )
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    image = image.resize((width, height))
    return image, height, width


def center_crop_resize(image, height, width):
    # Calculate resize ratio to match first frame dimensions
    resize_ratio = max(width / image.width, height / image.height)

    # Resize the image
    width = round(image.width * resize_ratio)
    height = round(image.height * resize_ratio)
    size = [width, height]
    image = TF.center_crop(image, size)

    return image, height, width


def prepare_pipeline(
    pipe: WanImageToVideoPipeline,
    args: argparse.ArgumentParser,
):
    if args.cache:
        cache_dit.enable_cache(
            pipe,
            # Cache context kwargs
            enable_separate_cfg=True,
            enable_taylorseer=True,
            enable_encoder_taylorseer=True,
            taylorseer_order=2,
        )

    # Enable memory savings
    pipe.enable_model_cpu_offload()

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

    return pipe


def main():
    args = get_args()
    print(args)

    model_id = os.environ.get(
        "WAN_FLF2V_DIR",
        "Wan-AI/Wan2.1-FLF2V-14B-720P-Diffusers",
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        model_id, subfolder="image_encoder", torch_dtype=torch.float32
    )
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        model_id,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")

    pipe = prepare_pipeline(pipe, args)

    first_frame = load_image("../data/flf2v_input_first_frame.png")
    last_frame = load_image("../data/flf2v_input_last_frame.png")

    first_frame, height, width = aspect_ratio_resize(first_frame, pipe)
    if last_frame.size != first_frame.size:
        last_frame, _, _ = center_crop_resize(last_frame, height, width)

    prompt = (
        "CG animation style, a small blue bird takes off from the ground, flapping its wings. "
        + "The bird's feathers are delicate, with a unique pattern on its chest. The background shows "
        + "a blue sky with white clouds under bright sunshine. The camera follows the bird upward, "
        + "capturing its flight and the vastness of the sky from a close-up, low-angle perspective."
    )

    start = time.time()
    output = pipe(
        image=first_frame,
        last_image=last_frame,
        prompt=prompt,
        height=height,
        width=width,
        guidance_scale=5.5,
        num_frames=49,
        num_inference_steps=35,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    end = time.time()

    stats = cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"wan.flf2v.{strify(args, stats)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(output, save_path, fps=16)


if __name__ == "__main__":
    main()
