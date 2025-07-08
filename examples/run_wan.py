import os
import time
import torch
import argparse
import diffusers
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=2)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=1)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--rdt", type=float, default=0.08)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--disable-cfg-diff-compute-separate",
        "--disable-cfg-ddiff",
        action="store_false",
        default=False,
    )
    return parser.parse_args()


args = get_args()
print(args)


height, width = 480, 832
pipe = WanPipeline.from_pretrained(
    os.environ.get(
        "WAN_DIR",
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",  # "num_layers": 30,
    ),
    torch_dtype=torch.bfloat16,
)

# flow shift should be 3.0 for 480p images, 5.0 for 720p images
if hasattr(pipe, "scheduler") and pipe.scheduler is not None:
    # Use the UniPCMultistepScheduler with the specified flow shift
    flow_shift = 3.0 if height == 480 else 5.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )

pipe.to("cuda")

if args.cache:
    cache_options = {
        "cache_type": CacheType.DBCache,
        "warmup_steps": args.warmup_steps,
        "max_cached_steps": -1,  # -1 means no limit
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
        "cfg_diff_compute_separate": not args.disable_cfg_diff_compute_separate,
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

    apply_cache_on_pipe(pipe, **cache_options)
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

start = time.time()
video = pipe(
    prompt=(
        "An astronaut dancing vigorously on the moon with earth "
        "flying past in the background, hyperrealistic"
    ),
    negative_prompt="",
    height=height,
    width=width,
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
save_path = f"wan.{cache_type_str}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=16)
