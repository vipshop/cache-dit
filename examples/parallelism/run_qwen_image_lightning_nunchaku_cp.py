import os
import sys

sys.path.append("..")

import time
import torch
import math
import torch.distributed as dist
from diffusers import (
    QwenImagePipeline,
    FlowMatchEulerDiscreteScheduler,
    PipelineQuantizationConfig,
)
from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel,
)
from utils import (
    get_args,
    strify,
    maybe_init_distributed,
    maybe_destroy_distributed,
    MemoryTracker,
)
import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

steps = 8 if args.steps is None else args.steps
assert steps in [8, 4]

nunchaku_qwen_image_dir = os.environ.get(
    "NUNCHAKA_QWEN_IMAGE_DIR",
    "nunchaku-tech/nunchaku-qwen-image",
)
lightning_version = "v1.1" if steps == 8 else "v1.0"
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"{nunchaku_qwen_image_dir}/svdq-int4_r32-qwen-image-lightning"
    f"{lightning_version}-{steps}steps.safetensors"
)

# Minimize VRAM required: 25GiB if use w4a16_text_encoder else 35GiB
w4a16_text_encoder = args.quantize
pipe = QwenImagePipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "QWEN_IMAGE_DIR",
            "Qwen/Qwen-Image",
        )
    ),
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
    quantization_config=(
        PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["text_encoder"],
        )
        if w4a16_text_encoder
        else None
    ),
).to("cuda")


if args.cache or args.parallel_type is not None:
    from cache_dit import (
        DBCacheConfig,
        ParallelismConfig,
        TaylorSeerCalibratorConfig,
    )

    cache_dit.enable_cache(
        pipe,
        cache_config=(
            DBCacheConfig(
                Fn_compute_blocks=16,
                Bn_compute_blocks=16,
                max_warmup_steps=4 if steps > 4 else 2,
                warmup_interval=2 if steps > 4 else 1,
                max_cached_steps=2 if steps > 4 else 1,
                max_continuous_cached_steps=1,
                enable_separate_cfg=False,  # true_cfg_scale=1.0
                residual_diff_threshold=0.50 if steps > 4 else 0.8,
            )
            if args.cache
            else None
        ),
        calibrator_config=(
            TaylorSeerCalibratorConfig(
                taylorseer_order=args.taylorseer_order,
            )
            if args.taylorseer
            else None
        ),
        parallelism_config=(
            ParallelismConfig(
                ulysses_size=(dist.get_world_size() if args.parallel_type == "ulysses" else None),
                ring_size=(dist.get_world_size() if args.parallel_type == "ring" else None),
            )
            if args.parallel_type in ["ulysses", "ring"]
            else None
        ),
    )


positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
}

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

if args.prompt is not None:
    prompt = args.prompt
# using an empty string if you do not have specific concept to remove
negative_prompt = " "
if args.negative_prompt is not None:
    negative_prompt = args.negative_prompt

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe():
    # do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    output = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=1024 if args.width is None else args.width,
        height=1024 if args.height is None else args.height,
        num_inference_steps=steps,
        true_cfg_scale=1.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output_type="latent" if args.perf else "pil",
    )
    image = output.images[0] if not args.perf else None
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

# warmup
_ = run_pipe()

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()


if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"qwen-image-lightning.{steps}steps.nunchaku.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    if not args.perf:
        print(f"Saving image to {save_path}")
        image.save(save_path)

maybe_destroy_distributed()
