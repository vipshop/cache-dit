import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from utils import (
    GiB,
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
    MemoryTracker,
)

import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get(
        "WAN_2_2_DIR",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    )
)
pipe = WanPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

if args.quantize:
    assert isinstance(args.quantize_type, str)
    if hasattr(pipe, "transformer"):
        if args.quantize_type.endswith("wo"):
            pipe.transformer = cache_dit.quantize(
                pipe.transformer,
                quant_type=args.quantize_type,
            )
    if hasattr(pipe, "transformer_2"):
        pipe.transformer_2 = cache_dit.quantize(
            pipe.transformer_2,
            quant_type=args.quantize_type,
        )
    print(f"Applied quantization: {args.quantize_type} to Wan transformers.")

if args.cache or args.parallel_type is not None:
    from cache_dit import (
        ForwardPattern,
        BlockAdapter,
        ParamsModifier,
        DBCacheConfig,
    )

    if "Wan2.1" in model_id:
        cachify(args, pipe)
    else:
        # Wan 2.2 only
        cachify(
            args,
            BlockAdapter(
                pipe=pipe,
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
        )

# Enable memory savings
if GiB() < 40:
    pipe.enable_model_cpu_offload(device=device)
else:
    pipe.to(device)

assert isinstance(pipe.transformer, WanTransformer3DModel)

pipe.set_progress_bar_config(disable=rank != 0)

# Set default prompt
prompt = "A cat walks on the grass, realistic"
if args.prompt is not None:
    prompt = args.prompt

negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
if args.negative_prompt is not None:
    negative_prompt = args.negative_prompt

height = 480 if args.height is None else args.height
width = 832 if args.width is None else args.width


def run_pipe(warmup: bool = False):
    seed = 1234
    generator = torch.Generator(device="cpu").manual_seed(seed)

    num_inference_steps = 30 if not warmup else 4
    if args.steps is not None and not warmup:
        num_inference_steps = args.steps
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=49,
        guidance_scale=5.0,
        generator=generator,
        num_inference_steps=num_inference_steps,
    ).frames[0]
    return output


if args.attn is not None:
    if hasattr(pipe.transformer, "set_attention_backend"):
        pipe.transformer.set_attention_backend(args.attn)
        print(f"Set attention backend to {args.attn}")


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)
    if hasattr(pipe, "transformer_2"):
        pipe.transformer_2 = torch.compile(pipe.transformer_2)

# warmup
_ = run_pipe(warmup=True)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
video = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"wan.{height}x{width}.{strify(args, pipe)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=16)

maybe_destroy_distributed()
