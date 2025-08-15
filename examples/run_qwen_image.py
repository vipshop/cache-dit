import os
import time
import torch
import argparse
from diffusers import QwenImagePipeline
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType
from utils import GiB


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    # General arguments
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--cache-type", "--type", type=str, default="dbcache")
    parser.add_argument("--taylorseer", action="store_true", default=False)
    parser.add_argument("--taylorseer-order", "--order", type=int, default=4)
    parser.add_argument("--Fn-compute-blocks", "--Fn", type=int, default=8)
    parser.add_argument("--Bn-compute-blocks", "--Bn", type=int, default=0)
    parser.add_argument("--rdt", type=float, default=0.12)
    parser.add_argument("--warmup-steps", type=int, default=8)
    return parser.parse_args()


args = get_args()
print(args)


pipe = QwenImagePipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_DIR",
        "Qwen/Qwen-Image",
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
    max_memory=(
        {i: f"{GiB()}GB" for i in range(torch.cuda.device_count())}
        if args.cache_type != "dbcache"
        else None
    ),
)


if args.cache:
    if args.cache_type.lower() == "dbcache":
        cache_options = {
            "cache_type": CacheType.DBCache,
            "warmup_steps": args.warmup_steps,
            "max_cached_steps": -1,  # -1 means no limit
            # Fn=1, Bn=0, means FB Cache, otherwise, Dual Block Cache
            "Fn_compute_blocks": args.Fn_compute_blocks,  # Fn, F8, etc.
            "Bn_compute_blocks": args.Bn_compute_blocks,  # Bn, B16, etc.
            "residual_diff_threshold": args.rdt,
            # CFG: classifier free guidance or not
            "do_separate_classifier_free_guidance": True,
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
            f"T{int(args.taylorseer)}O{args.taylorseer_order}_"
            f"R{args.rdt}"
        )
    elif args.cache_type.lower() == "dbprune":
        cache_options = {
            "cache_type": CacheType.DBPrune,
            "warmup_steps": args.warmup_steps,
            "max_pruned_steps": -1,  # -1 means no limit
            "Fn_compute_blocks": args.Fn_compute_blocks,  # Fn, F8, etc.
            "Bn_compute_blocks": args.Bn_compute_blocks,  # Bn, B16, etc.
            "residual_diff_threshold": args.rdt,
            # CFG: classifier free guidance or not
            "do_separate_classifier_free_guidance": True,
            "cfg_compute_first": False,
        }
        cache_type_str = "DBPRUNE"
        cache_type_str = (
            f"{cache_type_str}_F{args.Fn_compute_blocks}"
            f"B{args.Bn_compute_blocks}W{args.warmup_steps}_"
            f"R{args.rdt}"
        )
    else:
        raise ValueError("cache_type error, only support dbcache and dbprune.")

    print(f"cache options:\n{cache_options}")

    apply_cache_on_pipe(pipe, **cache_options)
else:
    cache_type_str = "NONE"


if torch.cuda.device_count() <= 1 or (
    args.cache and args.cache_type != "dbcache"
):
    # Enable memory savings
    pipe.reset_device_map()
    pipe.enable_model_cpu_offload()


positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
}

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

# using an empty string if you do not have specific concept to remove
negative_prompt = " "


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

start = time.time()

# do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

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
if hasattr(pipe.transformer, "_pruned_blocks"):
    pruned_steps = pipe.transformer._pruned_steps
    pruned_blocks = pipe.transformer._pruned_blocks
    actual_blocks = pipe.transformer._actual_blocks
    residual_diffs = pipe.transformer._residual_diffs
    pruned_ratio = (
        sum(pruned_blocks) / sum(actual_blocks) if actual_blocks else 0
    ) * 100
    print(
        f"Pruned Steps: {pruned_steps}, {pruned_blocks}, "
        f"Pruned Blocks: {sum(pruned_blocks)}({pruned_ratio:.2f})%"
    )
    print(f"Residual Diffs: {len(residual_diffs)}, {residual_diffs}")
if hasattr(pipe.transformer, "_cfg_pruned_blocks"):
    cfg_pruned_steps = pipe.transformer._cfg_pruned_steps
    cfg_pruned_blocks = pipe.transformer._cfg_pruned_blocks
    cfg_actual_blocks = pipe.transformer._cfg_actual_blocks
    cfg_residual_diffs = pipe.transformer._cfg_residual_diffs
    cfg_pruned_ratio = (
        sum(cfg_pruned_blocks) / sum(cfg_actual_blocks)
        if cfg_actual_blocks
        else 0
    ) * 100
    print(
        f"CFG Pruned Steps: {cfg_pruned_steps}, {cfg_pruned_blocks}, "
        f"CFG Pruned Blocks: {sum(cfg_pruned_blocks)}({cfg_pruned_ratio:.2f})%"
    )
    print(
        f"CFG Residual Diffs: {len(cfg_residual_diffs)}, {cfg_residual_diffs}"
    )


time_cost = end - start
save_path = f"qwen-image.{cache_type_str}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
