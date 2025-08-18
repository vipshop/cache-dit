import os
import time
import torch
import diffusers
import argparse
import numpy as np
import torchvision.transforms.functional as TF
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from transformers import CLIPVisionModel
import cache_dit


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=2)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=1)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--downsample-factor", "--df", type=int, default=4)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--warmup-steps", type=int, default=0)
    return parser.parse_args()


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
        cache_options = {
            "cache_type": cache_dit.DBCache,
            "warmup_steps": args.warmup_steps,
            "max_cached_steps": -1,  # -1 means no limit
            "downsample_factor": args.downsample_factor,
            # Fn=1, Bn=0, means FB Cache, otherwise, Dual Block Cache
            "Fn_compute_blocks": args.Fn_compute_blocks,  # Fn, F8, etc.
            "Bn_compute_blocks": args.Bn_compute_blocks,  # Bn, B16, etc.
            "residual_diff_threshold": args.rdt,
            # releative token diff threshold, default is 0.0
            "important_condition_threshold": 0.00,
            # CFG: classifier free guidance or not
            # For model that fused CFG and non-CFG into single forward step,
            # should set do_separate_classifier_free_guidance as False.
            "do_separate_classifier_free_guidance": True,
            # Compute cfg forward first or not, default False, namely,
            # 0, 2, 4, ..., -> non-CFG step; 1, 3, 5, ... -> CFG step.
            "cfg_compute_first": False,
            # Compute spearate diff values for CFG and non-CFG step,
            # default True. If False, we will use the computed diff from
            # current non-CFG transformer step for current CFG step.
            "cfg_diff_compute_separate": True,
            "enable_taylorseer": args.taylorseer,
            "enable_encoder_taylorseer": args.taylorseer,
            # Taylorseer cache type cache be hidden_states or residual
            "taylorseer_cache_type": "residual",
            "taylorseer_kwargs": {
                "n_derivatives": args.taylorseer_order,
            },
        }
        cache_type_str = "DBCACHE"
        cache_type_str = (
            f"{cache_type_str}_F{args.Fn_compute_blocks}"
            f"B{args.Bn_compute_blocks}W{args.warmup_steps}"
            f"T{int(args.taylorseer)}O{args.taylorseer_order}"
        )
        print(f"cache options:\n{cache_options}")

        cache_dit.enable_cache(pipe, **cache_options)
    else:
        cache_type_str = "NONE"

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

    return cache_type_str, pipe


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

    cache_type_str, pipe = prepare_pipeline(pipe, args)

    first_frame = load_image("data/flf2v_input_first_frame.png")
    last_frame = load_image("data/flf2v_input_last_frame.png")

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

    if hasattr(pipe.transformer, "_cached_steps"):
        cached_steps = pipe.transformer._cached_steps
        residual_diffs = pipe.transformer._residual_diffs
        print(f"Cache Steps: {len(cached_steps)}, {cached_steps}")
        print(f"Residual Diffs: {len(residual_diffs)}, {residual_diffs}")
    if hasattr(pipe.transformer, "_cfg_cached_steps"):
        cfg_cached_steps = pipe.transformer._cfg_cached_steps
        cfg_residual_diffs = pipe.transformer._cfg_residual_diffs
        print(f"CFG Cache Steps: {len(cfg_cached_steps)}, {cfg_cached_steps} ")
        print(
            f"CFG Residual Diffs: {len(cfg_residual_diffs)}, {cfg_residual_diffs}"
        )

    time_cost = end - start
    save_path = f"wan.flf2v.{cache_type_str}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(output, save_path, fps=16)


if __name__ == "__main__":
    main()
