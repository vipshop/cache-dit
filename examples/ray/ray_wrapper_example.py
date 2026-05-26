from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit import ParallelismConfig
from cache_dit import QuantizeConfig


def parse_args() -> argparse.Namespace:
  """Parse command line arguments for the Ray wrapper example.

  :returns: Parsed command line arguments.
  """

  parser = argparse.ArgumentParser(
    description="Run FLUX.2-klein-base-9B with optional cache-dit Ray wrapper.")
  parser.add_argument("--model-path",
                      type=str,
                      default=None,
                      help="Path to FLUX.2-klein-base-9B model.")
  parser.add_argument("--prompt", type=str, default="A cat holding a sign that says hello world")
  parser.add_argument("--height", type=int, default=1024)
  parser.add_argument("--width", type=int, default=1024)
  parser.add_argument("--num-inference-steps", "--steps", type=int, default=28)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--warmup",
                      type=int,
                      default=1,
                      help="Number of warmup generations before timing.")
  parser.add_argument("--repeat", type=int, default=1, help="Number of timed generations.")
  parser.add_argument(
    "--cache",
    action="store_true",
    help="Enable cache-dit with the default DBCacheConfig.",
  )
  parser.add_argument(
    "--quantize",
    action="store_true",
    help="Enable quantization with the default QuantizeConfig.",
  )
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
    default="pipeline",
    help="Enable Ray wrapper on pipe.transformer or on the pipeline object.",
  )
  parser.add_argument(
    "--use-flashpack-transfer",
    action="store_true",
    help="Use Diffusers serialization with use_flashpack=True for Ray pipeline snapshots.",
  )
  parser.add_argument(
    "--use-compile",
    "--compile",
    action="store_true",
    help="Compile the Ray-owned transformer after loading and parallelization.",
  )
  parser.add_argument(
    "--use-init-fn",
    action="store_true",
    help="Use ray_transfer_fn to initialize pipeline on each worker instead of serialization.",
  )
  return parser.parse_args()


def _make_ray_init_fn(model_path: str):
  """Return an initialize_fn that loads the pipeline on each Ray worker.

  Defined at module level so Ray cloudpickle can serialize it reliably. Uses a closure that captures
  only *model_path* (a short string), never large model objects.
  """

  def _init_fn():
    pipe = Flux2KleinPipeline.from_pretrained(
      model_path,
      torch_dtype=torch.bfloat16,
    ).to("cuda")
    return pipe

  return _init_fn


def main() -> None:
  """Run the Ray wrapper example and save the generated image."""

  args = parse_args()
  model_path = args.model_path or os.environ.get(
    "FLUX_2_KLEIN_BASE_9B_DIR",
    "black-forest-labs/FLUX.2-klein-base-9B",
  )
  use_ray = args.ulysses > 1 or args.tp > 1

  if args.use_init_fn:
    # With ray_transfer_fn, workers load the model independently;
    # the main process does not need a pipeline copy at all.
    pipe = None
  else:
    pipe = Flux2KleinPipeline.from_pretrained(
      model_path,
      torch_dtype=torch.bfloat16,
    )
    if not use_ray:
      pipe.to("cuda")

  cache_config = DBCacheConfig(Fn_compute_blocks=1) if args.cache else None
  quantize_config = QuantizeConfig(quant_type="float8_per_tensor") if args.quantize else None
  parallelism_config = None
  accelerate_enabled = use_ray or cache_config is not None or quantize_config is not None

  if use_ray:
    init_fn = None
    if args.use_init_fn:
      if args.target != "pipeline":
        raise ValueError(
          "--use-init-fn requires --target pipeline (transformer-level init-fn is not supported).")
      init_fn = _make_ray_init_fn(model_path)
    parallelism_config = ParallelismConfig(
      ulysses_size=args.ulysses if args.ulysses > 1 else None,
      tp_size=args.tp if args.tp > 1 else None,
      use_ray=True,
      ray_transfer_fn=init_fn,
      ray_use_flashpack=args.use_flashpack_transfer,
      ray_use_compile=args.use_compile,
    )

  if accelerate_enabled:
    if args.use_init_fn:
      # pipe_or_adapter defaults to None; enable_cache returns a callable RayPipelineEngine.
      pipe = cache_dit.enable_cache(
        cache_config=cache_config,
        parallelism_config=parallelism_config,
        quantize_config=quantize_config,
      )
    elif args.target == "pipeline":
      # NOTE: Will auto transfer to cuda inside by ray wrapper for
      # pipeline-level parallelism, so we keep the original pipeline
      # on CPU to avoid redundant GPU memory usage.
      cache_dit.enable_cache(
        pipe,
        cache_config=cache_config,
        parallelism_config=parallelism_config,
        quantize_config=quantize_config,
      )
    else:
      cache_dit.enable_cache(
        pipe.transformer,
        cache_config=cache_config,
        parallelism_config=parallelism_config,
        quantize_config=quantize_config,
      )
      if use_ray:
        # NOTE: Only the transformer is parallelized and transferred to GPU,
        # so we need to move the pipeline to GPU as well for the forward pass.
        pipe.to("cuda")

  if args.warmup < 0:
    raise ValueError("--warmup must be greater than or equal to 0.")
  if args.repeat < 1:
    raise ValueError("--repeat must be greater than or equal to 1.")

  def run_generation():
    generator = torch.Generator("cpu").manual_seed(args.seed)
    return pipe(
      prompt=args.prompt,
      height=args.height,
      width=args.width,
      num_inference_steps=args.num_inference_steps,
      generator=generator,
    ).images[0]

  # Call the pipeline as usual; No code changes are needed for
  # Ray parallelism to work.
  for _ in range(args.warmup):
    run_generation()

  start_time = time.time()
  image = None
  for _ in range(args.repeat):
    image = run_generation()
  elapsed = time.time() - start_time
  assert image is not None

  save_path = Path(args.save_path)
  save_path.parent.mkdir(parents=True, exist_ok=True)
  image.save(save_path)
  print(f"Warmup: {args.warmup}")
  print(f"Repeat: {args.repeat}")
  print(f"Total Inference Time: {elapsed:.2f}s")
  print(f"Average Inference Time: {elapsed / args.repeat:.2f}s")
  print(f"Saved image to {save_path}")


if __name__ == "__main__":
  main()
  # Example usage:
  # python3 ray_wrapper_example.py # baseline with no Ray parallelism
  # python3 ray_wrapper_example.py --ulysses 2 --save-path ray_ulysses2_output.png
  # python3 ray_wrapper_example.py --tp 2 --cache --quantize --save-path ray_tp2_output.png
