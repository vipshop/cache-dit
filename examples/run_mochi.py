import os
import time
import torch
import argparse
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
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
    return parser.parse_args()


args = get_args()
print(args)


pipe = MochiPipeline.from_pretrained(
    os.environ.get(
        "MOCHI_DIR",
        "genmo/mochi-1-preview",
    ),
    torch_dtype=torch.bfloat16,
).to("cuda")

if args.cache:
    cache_options = {
        "cache_type": CacheType.DBCache,
        "warmup_steps": args.warmup_steps,
        "max_cached_steps": -1,  # -1 means no limit
        # Fn=1, Bn=0, means FB Cache, otherwise, Dual Block Cache
        "Fn_compute_blocks": args.Fn_compute_blocks,  # Fn, F8, etc.
        "Bn_compute_blocks": args.Bn_compute_blocks,  # Bn, B16, etc.
        "residual_diff_threshold": args.rdt,
        # CFG: classifier free guidance or not
        # Mochi fused CFG and non-CFG into single forward step
        # so, we set do_separate_classifier_free_guidance as False.
        "do_separate_classifier_free_guidance": False,
        "cfg_compute_first": False,
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

pipe.enable_vae_tiling()

prompt = (
    "Close-up of a chameleon's eye, with its scaly skin "
    "changing color. Ultra high resolution 4k."
)

start = time.time()
video = pipe(
    prompt,
    num_frames=84,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

time_cost = end - start
save_path = f"mochi.{cache_type_str}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=30)
