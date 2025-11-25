import os
import sys

sys.path.append("..")

import time
import torch
import PIL.Image
from diffusers import AutoencoderKLWan, WanVACEPipeline
from diffusers.schedulers.scheduling_unipc_multistep import (
    UniPCMultistepScheduler,
)
from diffusers.utils import export_to_video, load_image

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


def prepare_video_and_mask(
    first_img: PIL.Image.Image,
    last_img: PIL.Image.Image,
    height: int,
    width: int,
    num_frames: int,
):
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = []
    frames.append(first_img)
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask


model_id = args.model_path if args.model_path is not None else "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
model_id = (
    args.model_path if args.model_path is not None else os.environ.get("WAN_VACE_DIR", model_id)
)
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
pipe = WanVACEPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

torch.cuda.empty_cache()
# Enable memory savings
if GiB() < 40:
    pipe.enable_model_cpu_offload(device=device)
else:
    pipe.to(device)

# Add quantization support
if args.quantize:
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
        quant_type=args.quantize_type,
    )

assert isinstance(pipe.vae, AutoencoderKLWan)  # enable type check for IDE
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# Set default prompt and negative prompt
prompt = (
    "CG animation style, a small blue bird takes off from the ground, "
    "flapping its wings. The bird's feathers are delicate, with a unique "
    "pattern on its chest. The background shows a blue sky with white "
    "clouds under bright sunshine. The camera follows the bird upward, "
    "capturing its flight and the vastness of the sky from a close-up, "
    "low-angle perspective."
)
if args.prompt is not None:
    prompt = args.prompt

negative_prompt = (
    "Bright tones, overexposed, static, blurred details, subtitles, "
    "style, works, paintings, images, static, overall gray, worst "
    "quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
    "disfigured, misshapen limbs, fused fingers, still picture, messy "
    "background, three legs, many people in the background, walking "
    "backwards"
)
if args.negative_prompt is not None:
    negative_prompt = args.negative_prompt
first_frame = load_image("../data/flf2v_input_first_frame.png")
last_frame = load_image("../data/flf2v_input_last_frame.png")

height = 512
width = 512
num_frames = 81
video, mask = prepare_video_and_mask(first_frame, last_frame, height, width, num_frames)

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe(warmup: bool = False):
    output = pipe(
        video=video,
        mask=mask,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=30 if not warmup else 5,
        guidance_scale=5.0,
        generator=torch.Generator("cpu").manual_seed(42),
    ).frames[0]
    return output


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

# warmup
_ = run_pipe(warmup=True)


memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
output = run_pipe(warmup=False)
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

if rank == 0:

    stats = cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"wan-vace.{strify(args, stats)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(output, save_path, fps=16)

maybe_destroy_distributed()
