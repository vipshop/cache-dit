import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)


pipe = LTXConditionPipeline.from_pretrained(
    os.environ.get("LTX_VIDEO", "Lightricks/LTX-Video-0.9.7-dev"),
    torch_dtype=torch.bfloat16,
    quantization_config=PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["transformer", "text_encoder"],
    ),
)

pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
    os.environ.get("LTX_UPSCALER", "Lightricks/ltxv-spatial-upscaler-0.9.7"),
    vae=pipe.vae,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
pipe_upsample.to("cuda")
pipe.vae.enable_tiling()

if args.cache:
    cache_dit.enable_cache(
        pipe,
        Fn_compute_blocks=args.Fn,
        Bn_compute_blocks=args.Bn,
        max_warmup_steps=args.max_warmup_steps,
        max_cached_steps=args.max_cached_steps,
        max_continuous_cached_steps=args.max_continuous_cached_steps,
        enable_taylorseer=args.taylorseer,
        enable_encoder_taylorseer=args.taylorseer,
        taylorseer_order=args.taylorseer_order,
        residual_diff_threshold=args.rdt,
    )


def round_to_nearest_resolution_acceptable_by_vae(height, width):
    height = height - (height % pipe.vae_spatial_compression_ratio)
    width = width - (width % pipe.vae_spatial_compression_ratio)
    return height, width


prompt = "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
negative_prompt = (
    "worst quality, inconsistent motion, blurry, jittery, distorted"
)
expected_height, expected_width = 512, 704
downscale_factor = 2 / 3
num_frames = 121

# Part 1. Generate video at smaller resolution
downscaled_height, downscaled_width = int(
    expected_height * downscale_factor
), int(expected_width * downscale_factor)
downscaled_height, downscaled_width = (
    round_to_nearest_resolution_acceptable_by_vae(
        downscaled_height, downscaled_width
    )
)

start = time.time()
latents = pipe(
    conditions=None,
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=downscaled_width,
    height=downscaled_height,
    num_frames=num_frames,
    num_inference_steps=30,
    generator=torch.Generator("cpu").manual_seed(0),
    output_type="latent",
).frames
end = time.time()
stats = cache_dit.summary(pipe)

# Part 2. Upscale generated video using latent upsampler with fewer inference steps
# The available latent upsampler upscales the height/width by 2x
upscaled_height, upscaled_width = downscaled_height * 2, downscaled_width * 2
upscaled_latents = pipe_upsample(latents=latents, output_type="latent").frames

# Part 3. Denoise the upscaled video with few steps to improve texture (optional, but recommended)
video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=upscaled_width,
    height=upscaled_height,
    num_frames=num_frames,
    denoise_strength=0.4,  # Effectively, 4 inference steps out of 10
    num_inference_steps=10,
    latents=upscaled_latents,
    decode_timestep=0.05,
    image_cond_noise_scale=0.025,
    generator=torch.Generator("cpu").manual_seed(0),
    output_type="pil",
).frames[0]


# Part 4. Downscale the video to the expected resolution
video = [frame.resize((expected_width, expected_height)) for frame in video]

time_cost = end - start
save_path = f"ltx-video.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=8)
