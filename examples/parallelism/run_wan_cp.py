import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import WanPipeline, WanTransformer3DModel
from diffusers.utils import export_to_video
from utils import (
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

model_id = os.environ.get(
    "WAN_2_2_DIR",
    "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
)
pipe = WanPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

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

assert isinstance(pipe.transformer, WanTransformer3DModel)
pipe.enable_model_cpu_offload(device=device)

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(warmup: bool = False):
    prompt = "A cat walks on the grass, realistic"
    negative_prompt = "Bright tones, overexposed, static, blurred details, "
    "subtitles, style, works, paintings, images, static, overall gray, "
    "worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
    "disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking backwards"

    seed = 1234
    generator = torch.Generator(device="cpu").manual_seed(seed)

    num_inference_steps = 30 if not warmup else 4
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_frames=49,
        guidance_scale=5.0,
        generator=generator,
        num_inference_steps=num_inference_steps,
    ).frames[0]
    return output


# warmup
_ = run_pipe(warmup=True)

start = time.time()
video = run_pipe()
end = time.time()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"wan.{strify(args, pipe)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    export_to_video(video, save_path, fps=16)

maybe_destroy_distributed()
