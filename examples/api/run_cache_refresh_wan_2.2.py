import os
import sys

sys.path.append("..")

import time
import torch
import diffusers
from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from utils import get_args, GiB, strify, MemoryTracker
import cache_dit
from cache_dit.platforms import current_platform

parser = get_args(parse=False)
parser.add_argument(
    "--summary",
    action="store_true",
    default=False,
    help="Print summary of the model after each inference",
)
args = parser.parse_args()
print(args)


height, width = 480, 832
pipe = WanPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "WAN_2_2_DIR",
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        )
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=("balanced" if (current_platform.device_count() > 1 and GiB() <= 48) else None),
)

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
    # Use the UniPCMultistepScheduler with the specified flow shift
    flow_shift = 3.0 if height == 480 else 5.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )


if args.cache:
    from cache_dit import (
        BlockAdapter,
        ForwardPattern,
        ParamsModifier,
        DBCacheConfig,
    )

    assert isinstance(pipe.transformer, WanTransformer3DModel)
    assert isinstance(pipe.transformer_2, WanTransformer3DModel)

    # Dual transformer caching with transformer-only api in cache-dit.
    cache_dit.enable_cache(
        BlockAdapter(
            transformer=[
                pipe.transformer,
                pipe.transformer_2,
            ],
            blocks=[
                pipe.transformer.blocks,
                pipe.transformer_2.blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_2,
                ForwardPattern.Pattern_2,
            ],
            params_modifiers=[
                # high-noise transformer only have 30% steps
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=4,
                        max_cached_steps=8,
                    ),
                ),
                ParamsModifier(
                    cache_config=DBCacheConfig().reset(
                        max_warmup_steps=2,
                        max_cached_steps=20,
                    ),
                ),
            ],
            has_separate_cfg=True,
        ),
        cache_config=DBCacheConfig(
            Fn_compute_blocks=args.Fn,
            Bn_compute_blocks=args.Bn,
            max_warmup_steps=args.max_warmup_steps,
            max_cached_steps=args.max_cached_steps,
            max_continuous_cached_steps=args.max_continuous_cached_steps,
            residual_diff_threshold=args.rdt,
            # NOTE: num_inference_steps can be None here, we will
            # set it properly during cache refreshing.
            num_inference_steps=None,
        ),
    )

# When device_map is None, we need to explicitly move the model to GPU
# or enable CPU offload to avoid running on CPU
if current_platform.device_count() <= 1:
    # Single GPU: use CPU offload for memory efficiency
    pipe.enable_model_cpu_offload()
elif current_platform.device_count() > 1 and pipe.device.type == "cpu":
    # Multi-GPU but model is on CPU (device_map was None): move to default GPU
    pipe.to(current_platform.device_type)

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

assert isinstance(pipe.transformer, WanTransformer3DModel)
assert isinstance(pipe.transformer_2, WanTransformer3DModel)

if args.quantize:
    assert isinstance(args.quantize_type, str)
    if args.quantize_type.endswith("wo"):  # weight only
        pipe.transformer = cache_dit.quantize(
            pipe.transformer,
            quant_type=args.quantize_type,
        )
    # We only apply activation quantization (default: FP8 DQ)
    # for low-noise transformer to avoid non-trivial precision
    # downgrade.
    pipe.transformer_2 = cache_dit.quantize(
        pipe.transformer_2,
        quant_type=args.quantize_type,
    )

# Set default prompt and negative prompt
prompt = (
    "An astronaut dancing vigorously on the moon with earth "
    "flying past in the background, hyperrealistic"
)
if args.prompt is not None:
    prompt = args.prompt

negative_prompt = ""
if args.negative_prompt is not None:
    negative_prompt = args.negative_prompt


def split_inference_steps(num_inference_steps: int = 30) -> tuple[int, int]:
    if pipe.config.boundary_ratio is not None:
        boundary_timestep = pipe.config.boundary_ratio * pipe.scheduler.config.num_train_timesteps
    else:
        boundary_timestep = None
    pipe.scheduler.set_timesteps(num_inference_steps, device=current_platform.device_type)
    timesteps = pipe.scheduler.timesteps
    num_high_noise_steps = 0  # high-noise steps for transformer
    for t in timesteps:
        if boundary_timestep is not None and t >= boundary_timestep:
            num_high_noise_steps += 1
    # low-noise steps for transformer_2
    num_low_noise_steps = num_inference_steps - num_high_noise_steps
    return num_high_noise_steps, num_low_noise_steps


def run_pipe(steps: int = 30):

    if args.cache:
        # Refresh cache context with proper num_inference_steps
        num_high_noise_steps, num_low_noise_steps = split_inference_steps(
            num_inference_steps=steps,
        )

        cache_dit.refresh_context(
            pipe.transformer,
            num_inference_steps=num_high_noise_steps,
            verbose=True,
        )
        cache_dit.refresh_context(
            pipe.transformer_2,
            num_inference_steps=num_low_noise_steps,
            verbose=True,
        )

    gen_device = (
        current_platform.device_type if current_platform.is_accelerator_available() else "cpu"
    )
    video = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=81,
        num_inference_steps=steps,
        generator=torch.Generator(gen_device).manual_seed(0),
    ).frames[0]
    return video


if args.compile or args.quantize:
    cache_dit.set_compile_configs()
    pipe.transformer.compile_repeated_blocks(fullgraph=True)
    pipe.transformer_2.compile_repeated_blocks(fullgraph=True)

    # warmup
    run_pipe(steps=8)


memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()


steps = [16, 28, 50]
for i in range(len(steps)):
    print("-" * 150)
    start = time.time()
    video = run_pipe(steps=steps[i])
    end = time.time()
    time_cost = end - start

    save_path = f"wan2.2.steps{steps[i]}.{strify(args, pipe.transformer)}.mp4"
    export_to_video(video, save_path, fps=16)

    if args.summary:
        cache_dit.summary(pipe, details=True)
        print(f"Time cost: {time_cost:.2f}s")
        print(f"Saving video to {save_path}")


if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()
