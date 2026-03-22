"""torchrun --nproc_per_node=1 examples/api/test_dynamic_sp_1gpu_flux.py"""

import argparse
import os

import torch
from diffusers import FluxPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1-GPU validation for dynamic SP with a real FLUX model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get(
            "FLUX_DIR",
            "/project/infattllm/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9/",
        ),
        help="HF model id or local model path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/configs/dynamic_sp_1gpu.yaml",
        help="YAML config path containing parallelism_config.dynamic_sp.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A small robot reading a book in a cozy library, cinematic lighting.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=str,
        default="dynamic_sp_1gpu_flux.png",
        help="Output image path (saved on rank 0 only).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="For FLUX schnell, guidance_scale=0 is a safe default.",
    )
    parser.add_argument(
        "--compare-image",
        type=str,
        default=None,
        help="Optional reference image path (e.g. dynamic_sp_2gpu_flux.png) to compare pixel-wise.",
    )
    parser.add_argument(
        "--compare-tol",
        type=float,
        default=0.0,
        help="Max abs pixel diff tolerated for --compare-image (on 8-bit PNG data).",
    )
    return parser.parse_args()


def _assert_world_size(world_size: int, expected: int):
    if world_size != expected:
        raise RuntimeError(
            f"This validation script expects {expected} processes, but got world_size={world_size}. "
            f"Please launch with: torchrun --nproc_per_node={expected} ..."
        )


def main():
    args = parse_args()

    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    output = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="np",
    )
    image_np = output.images[0]

    pil_image = pipe.numpy_to_pil(image_np[None, ...])[0]
    pil_image.save(args.output)
    print(f"saved image: {args.output}")


if __name__ == "__main__":
    main()
