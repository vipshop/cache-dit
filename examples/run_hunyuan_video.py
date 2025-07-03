# Adapted from: https://github.com/chengzeyi/ParaAttention/blob/main/first_block_cache_examples/run_hunyuan_video.py
import os
import time
import torch
import argparse
from diffusers.utils import export_to_video
from diffusers import (
    HunyuanVideoPipeline,
    HunyuanVideoTransformer3DModel,
    AutoencoderKLHunyuanVideo,
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
    return parser.parse_args()


args = get_args()
print(args)


def get_gpu_memory_in_gib():
    if not torch.cuda.is_available():
        return 0

    try:
        total_memory_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device(),
        ).total_memory
        total_memory_gib = total_memory_bytes / (1024**3)
        return int(total_memory_gib)
    except Exception:
        return 0


model_id = os.environ.get("HUNYUAN_DIR", "tencent/HunyuanVideo")
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    revision="refs/pr/18",
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.float16,
    revision="refs/pr/18",
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
        # For model that fused CFG and non-CFG into single forward step,
        # should set do_separate_classifier_free_guidance as False.
        # NOTE: set it as True if true_cfg_scale > 1 and has_neg_prompt
        # for HunyuanVideoPipeline.
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

assert isinstance(
    pipe.vae, AutoencoderKLHunyuanVideo
)  # enable type check for IDE

# Enable memory savings
pipe.enable_model_cpu_offload()
if get_gpu_memory_in_gib() <= 48:
    pipe.vae.enable_tiling(
        # Make it runnable on GPUs with 48GB memory
        tile_sample_min_height=128,
        tile_sample_stride_height=96,
        tile_sample_min_width=128,
        tile_sample_stride_width=96,
        tile_sample_min_num_frames=32,
        tile_sample_stride_num_frames=24,
    )
else:
    pipe.vae.enable_tiling()


start = time.time()
output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=720,
    width=1280,
    num_frames=129,
    num_inference_steps=30,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

time_cost = end - start
save_path = f"hunyuan_video.{cache_type_str}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(output, save_path, fps=15)
