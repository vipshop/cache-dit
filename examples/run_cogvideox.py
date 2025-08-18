import os
import time
import torch
import argparse
from diffusers.utils import export_to_video
from diffusers import CogVideoXPipeline, AutoencoderKLCogVideoX
from utils import GiB
import cache_dit


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
    return parser.parse_args()


args = get_args()
print(args)


model_id = os.environ.get("COGVIDEOX_DIR", "THUDM/CogVideoX-5b")


def is_cogvideox_1_5():
    return "CogVideoX1.5" in model_id or "THUDM/CogVideoX1.5" in model_id


pipe = CogVideoXPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
).to("cuda")


if args.cache:
    cache_options = {
        "cache_type": cache_dit.DBCache,
        "warmup_steps": args.warmup_steps,
        "max_cached_steps": -1,  # -1 means no limit
        # Fn=1, Bn=0, means FB Cache, otherwise, Dual Block Cache
        "Fn_compute_blocks": args.Fn_compute_blocks,  # Fn, F8, etc.
        "Bn_compute_blocks": args.Bn_compute_blocks,  # Bn, B16, etc.
        "residual_diff_threshold": args.rdt,
        # releative token diff threshold, default is 0.0
        "important_condition_threshold": 0.05,
        # CFG: classifier free guidance or not
        # CogVideoX fused CFG and non-CFG into single forward step
        # so, we set do_separate_classifier_free_guidance as False.
        "do_separate_classifier_free_guidance": False,
        "cfg_compute_first": False,
        "enable_taylorseer": args.taylorseer,
        "enable_encoder_taylorseer": args.taylorseer,
        # Taylorseer cache type cache be hidden_states or residual
        "taylorseer_cache_type": "hidden_states",
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


pipe.enable_model_cpu_offload()
assert isinstance(pipe.vae, AutoencoderKLCogVideoX)  # enable type check for IDE
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

start = time.time()
prompt = (
    "A panda, dressed in a small, red jacket and a tiny hat, "
    "sits on a wooden stool in a serene bamboo forest. The "
    "panda's fluffy paws strum a miniature acoustic guitar, "
    "producing soft, melodic tunes. Nearby, a few other pandas "
    "gather, watching curiously and some clapping in rhythm. "
    "Sunlight filters through the tall bamboo, casting a gentle "
    "glow on the scene. The panda's face is expressive, showing "
    "concentration and joy as it plays. The background includes "
    "a small, flowing stream and vibrant green foliage, enhancing "
    "the peaceful and magical atmosphere of this unique musical "
    "performance."
)
video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=(
        # Avoid OOM for CogVideoX1.5 model on 48GB GPU
        16
        if (is_cogvideox_1_5() and GiB() <= 48)
        else 49
    ),
    guidance_scale=6,
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
save_path = f"cogvideox.{cache_type_str}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=8)
