from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

import cache_dit
from cache_dit import ParallelismConfig
from cache_dit.platforms import current_platform


def parse_args() -> argparse.Namespace:
  """Parse command line arguments for the Ray wrapper example.

  :returns: Parsed command line arguments.
  """

  parser = argparse.ArgumentParser(
    description="Run FLUX.2-klein-9B with optional cache-dit Ray wrapper.")
  parser.add_argument("--model-path", type=str, default=None, help="Path to FLUX.2-klein-9B model.")
  parser.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
  parser.add_argument("--height", type=int, default=1024)
  parser.add_argument("--width", type=int, default=1024)
  parser.add_argument("--num-inference-steps", "--steps", type=int, default=4)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument(
    "--ulysses",
    type=int,
    default=1,
    help="Ulysses size. Values > 1 enable Ray.",
  )
  parser.add_argument(
    "--tp",
    type=int,
    default=1,
    help="Tensor parallel size. Values > 1 enable Ray tensor parallelism.",
  )
  parser.add_argument("--save-path", type=str, default=".tmp/ray_wrapper.png")
  parser.add_argument(
    "--target",
    choices=("transformer", "pipeline"),
    default="transformer",
    help="Enable Ray wrapper on pipe.transformer or on the pipeline object.",
  )
  parser.add_argument(
    "--use-flashpack-transfer",
    action="store_true",
    help=
    "Use Diffusers save_pretrained/from_pretrained with use_flashpack=True for Ray pipeline snapshots.",
  )
  parser.add_argument(
    "--use-compile",
    action="store_true",
    help="Compile the Ray-owned transformer after loading and parallelization.",
  )
  return parser.parse_args()


def main() -> None:
  """Run the Ray wrapper example and save the generated image."""

  args = parse_args()
  model_path = args.model_path or os.environ.get(
    "FLUX_2_KLEIN_9B_DIR",
    "/workspace/dev/vipdev/hf_models/FLUX.2-klein-9B",
  )
  use_ray = args.ulysses > 1 or args.tp > 1
  pipe = Flux2KleinPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
  )

  if not use_ray:
    pipe.to(current_platform.device_type)

  if use_ray:
    parallelism_config = ParallelismConfig(
      ulysses_size=args.ulysses if args.ulysses > 1 else None,
      tp_size=args.tp if args.tp > 1 else None,
      use_ray=True,
      ray_use_flashpack=args.use_flashpack_transfer,
      ray_use_compile=args.use_compile,
    )
    if args.target == "pipeline":
      cache_dit.enable_cache(pipe, parallelism_config=parallelism_config)
    else:
      cache_dit.enable_cache(pipe.transformer, parallelism_config=parallelism_config)
      pipe.to(current_platform.device_type)

  generator = torch.Generator("cpu").manual_seed(args.seed)
  start_time = time.time()
  image = pipe(
    prompt=args.prompt,
    height=args.height,
    width=args.width,
    num_inference_steps=args.num_inference_steps,
    generator=generator,
  ).images[0]
  elapsed = time.time() - start_time

  save_path = Path(args.save_path)
  save_path.parent.mkdir(parents=True, exist_ok=True)
  image.save(save_path)
  print(f"Inference Time: {elapsed:.2f}s")
  print(f"Saved image to {save_path}")

  if use_ray:
    cache_dit.disable_cache(pipe if args.target == "pipeline" else pipe.transformer)


if __name__ == "__main__":
  main()
