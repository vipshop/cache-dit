"""Profile registered pipelines without adding instrumentation to the generate CLI."""

import importlib
import time
from contextlib import ExitStack, nullcontext

import numpy as np
import torch

from cache_dit import ProfilerContext, summary
from cache_dit.cli.registers import (
  ExampleOutputData,
  ExampleRegister,
  ExampleType,
  _AUDIO_TASKS,
  _AUDIO_VIDEO_TASKS,
  _VIDEO_TASKS,
  parse_audio_output,
  parse_audio_video_output,
)
from cache_dit.cli.utils import (
  MemoryTracker,
  get_args as get_base_args,
  maybe_apply_optimization,
  maybe_destroy_distributed,
  maybe_finalize_deferred_svdq_pipe_move,
  maybe_postprocess_args,
  strify,
)
from cache_dit.logger import init_logger
from cache_dit.platforms import current_platform
from cache_dit.profile.nvtx import LayerwiseNvtxMarker, nvtx_scope
from cache_dit.utils import iter_transformer_modules

logger = init_logger(__name__)


def get_args(parse=True):
  """Extend the existing model parser with standalone profiling options.

  :param parse: Parse command-line arguments or return the parser.
  :returns: Processed arguments or an independent parser.
  """
  parser = get_base_args(parse=False)
  parser.add_argument("example", choices=ExampleRegister.list_examples())
  parser.add_argument(
    "--example-factory",
    help="Optional module:function that creates an Example from the parsed arguments",
  )
  parser.add_argument(
    "--profile-collector",
    choices=["torch", "nsys"],
    default="torch",
    help="Use Torch export or CUDA gates for an external Nsight Systems session",
  )
  parser.add_argument(
    "--profile-ranges",
    default="all",
    help="Whole request ('all') or inclusive denoising steps (e.g. '0-2,8-10')",
  )
  parser.add_argument(
    "--no-profile-with-stack",
    dest="profile_with_stack",
    action="store_false",
    help="Collect an operator/kernel timeline without Python stacks",
  )
  parser.add_argument(
    "--nvtx",
    action="store_true",
    help="Enable stage NVTX annotations after warmup, independently of --profile",
  )
  parser.add_argument(
    "--layerwise-nvtx",
    action="store_true",
    help="Enable stage and transformer module NVTX annotations after warmup",
  )
  return maybe_postprocess_args(parser.parse_args()) if parse else parser


def run(example):
  """Own the profiling lifecycle while reusing the example's model and media helpers.

  :param example: Registered example with its ordinary initialization and input configuration.
  """
  args = example.args
  try:
    example.check_valid()
    input_kwargs, extra_optimize_kwargs = example.prepare_input_data()
    start_time = time.time()
    pipe = example.init_config.get_pipe(args)
    load_time = time.time() - start_time
    maybe_apply_optimization(args, pipe, **extra_optimize_kwargs)
    pipe.set_progress_bar_config(disable=example.rank != 0)

    with (MemoryTracker() if args.track_memory else nullcontext()) as memory_tracker:
      warmup_time = _warmup(example, pipe, input_kwargs)
      with ExitStack() as annotations:
        if args.layerwise_nvtx:
          annotations.enter_context(LayerwiseNvtxMarker(dict(iter_transformer_modules(pipe))))
        elif args.nvtx:
          annotations.enter_context(nvtx_scope())

        start_time = time.time()
        for _ in range(args.repeat):
          with (ProfilerContext(
              activities=args.profile_activities,
              output_dir=args.profile_dir,
              profile_name=args.profile_name or f"{args.example}_profile",
              with_stack=args.profile_with_stack,
              record_shapes=args.profile_record_shapes,
              collector=args.profile_collector,
              ranges=args.profile_ranges,
              devices=(list(range(current_platform.device_count()))
                       if args.device_map_balance else [example.device]),
          ) if args.profile else nullcontext()) as profiler:
            step_kwargs = example.new_generator(dict(input_kwargs), args)
            output = example.call_pipe(pipe, step_kwargs)
            maybe_finalize_deferred_svdq_pipe_move(pipe)
          if args.profile and example.rank == 0:
            if not profiler.results:
              logger.info("Profiler produced no windows: %s", profiler.empty_reason)
            for result in profiler.results:
              logger.info(
                "Profiler loop=%s window=%s requested=%s actual=%s status=%s "
                "last_step_complete=%s trace=%s memory=%s summary=%s", result.loop_id,
                result.window_id, result.requested_range, result.actual_range, result.status,
                result.last_step_complete, result.trace_path, result.memory_snapshot_path,
                result.memory_summary_path)
        inference_time = (time.time() - start_time) / args.repeat if args.repeat > 0 else None

      if args.cache_summary and example.rank == 0:
        summary(pipe)

    _save_output(
      example, pipe, output,
      ExampleOutputData(
        model_tag=args.example,
        strify_tag=strify(args, pipe),
        load_time=load_time,
        warmup_time=warmup_time,
        inference_time=inference_time,
        memory_usage=memory_tracker.report() if memory_tracker else None,
      ))
  finally:
    maybe_destroy_distributed()


def _warmup(example, pipe, input_kwargs):
  """Keep the example's warmup inputs and deferred device move unchanged.

  :param example: Registered example.
  :param pipe: Loaded and optimized pipeline.
  :param input_kwargs: Formal inference inputs.
  :returns: Average warmup duration, or None when warmup is disabled.
  """
  args = example.args
  start_time = time.time()
  for _ in range(args.warmup):
    warmup_kwargs = example.new_generator(dict(input_kwargs), args, warmup=True)
    if args.warmup_num_inference_steps is not None:
      warmup_kwargs["num_inference_steps"] = args.warmup_num_inference_steps
    example.call_pipe(pipe, warmup_kwargs)
    maybe_finalize_deferred_svdq_pipe_move(pipe)
  return (time.time() - start_time) / args.warmup if args.warmup > 0 else None


def _save_output(example, pipe, output, output_data):
  """Reuse the CLI's media parsers and output writer for the final result.

  :param example: Registered example.
  :param pipe: Pipeline that produced the output.
  :param output: Raw pipeline output.
  :param output_data: Timing and memory metadata.
  """
  task = example.init_config.task_type
  if task in (ExampleType.T2I, ExampleType.IE2I):
    images = output.images if hasattr(output, "images") else output
    if isinstance(images, list) and not example.init_config.preserve_image_list:
      images = images[0]
    output_data.image = images
  elif task in _AUDIO_VIDEO_TASKS:
    parsed = parse_audio_video_output(output,
                                      task,
                                      native_fps=getattr(pipe, "fps", None),
                                      pipe=pipe)
    output_data.video = parsed["video"]
    output_data.audio = parsed["audio"]
    output_data.audio_sampling_rate = parsed["audio_sampling_rate"]
    output_data.video_fps = parsed["video_fps"]
  elif task in _AUDIO_TASKS:
    parsed = parse_audio_output(output, task, pipe=pipe)
    output_data.audio = parsed["audio"]
    output_data.audio_sampling_rate = parsed["audio_sampling_rate"]
  elif task in _VIDEO_TASKS:
    if isinstance(output, dict):
      videos = output.get("videos")
      if videos is None:
        videos = output.get("frames", output)
    elif getattr(output, "videos", None) is not None:
      videos = output.videos
    else:
      videos = getattr(output, "frames", output)
    # Unwrap one batch level, keeping the CLI's tensor/list distinction.
    if torch.is_tensor(videos) or isinstance(videos, np.ndarray):
      if videos.ndim == 5:
        videos = videos[0]
    elif isinstance(videos, (list, tuple)):
      videos = videos[0]
    output_data.video = videos

  example.output_data = output_data
  if example.rank == 0:
    example.init_config.summary(example.args,
                                extra_model_path=ExampleRegister.get_default(example.args.example))
    example.input_data.summary(example.args)
    output_data.summary(example.args)
    output_data.save(example.args)


def main():
  args = get_args()
  if args.example_factory:
    module, name = args.example_factory.split(":")
    example = getattr(importlib.import_module(module), name)(args)
  else:
    example = ExampleRegister.get_example(args, args.example)
  run(example)


if __name__ == "__main__":
  main()
