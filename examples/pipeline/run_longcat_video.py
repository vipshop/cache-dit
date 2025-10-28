import os
import sys

sys.path.append("..")

import time
import datetime
import numpy as np

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from torchvision.io import write_video

sys.path.append(os.environ.get("LONGCAT_VIDEO_PKG_DIR", ""))
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import (
    LongCatVideoTransformer3DModel,
)
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import (
    init_context_parallel,
)

from utils import get_args, strify
import cache_dit

# Example usage:
# export LONGCAT_VIDEO_PKG_DIR=/path/to/codes/of/LongCat-Video
# export LONGCAT_VIDEO_DIR=/path/to/models/of/LongCat-Video
# Add `--quantize` to enable loading models with bitsandbytes
# for lower memory usage (e.g, GPU w/ < 48GB memory)
# torchrun --nproc_per_node=4 run_longcat_video.py --quantize --compile
# torchrun --nproc_per_node=4 run_longcat_video.py --quantize --compile --cache
# torchrun --nproc_per_node=4 run_longcat_video.py --quantize --compile --cache --Fn 1


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate(args):
    print(args)
    # case setup
    prompt = "In a realistic photography style, a white boy around seven or eight years old sits on a park bench, wearing a light blue T-shirt, denim shorts, and white sneakers. He holds an ice cream cone with vanilla and chocolate flavors, and beside him is a medium-sized golden Labrador. Smiling, the boy offers the ice cream to the dog, who eagerly licks it with its tongue. The sun is shining brightly, and the background features a green lawn and several tall trees, creating a warm and loving scene."
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    # load parsed args
    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size

    # prepare distributed environment
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24)
    )
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    if context_parallel_size is None:
        context_parallel_size = num_processes

    # initialize context parallel before loading models
    init_context_parallel(
        context_parallel_size=context_parallel_size,
        global_rank=global_rank,
        world_size=num_processes,
    )
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )

    # Load text encoder with bnb 4bits quantization if specified
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        quantization_config=(
            TransformersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if args.quantize
            else None
        ),
    )

    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
    )

    # Load DiT with bnb 4bits/8bits quantization if specified
    if args.quantize:
        if context_parallel_size >= 4:
            # Activation will be split across multiple GPUs in CP,
            # so we only apply FP8 Weight Only Quantization here
            # to keep higher accuracy.
            dit = LongCatVideoTransformer3DModel.from_pretrained(
                checkpoint_dir,
                subfolder="dit",
                cp_split_hw=cp_split_hw,
                torch_dtype=torch.bfloat16,
            )
            dit = cache_dit.quantize(dit, quant_type="fp8_w8a16_wo")
        else:
            dit = LongCatVideoTransformer3DModel.from_pretrained(
                checkpoint_dir,
                subfolder="dit",
                cp_split_hw=cp_split_hw,
                torch_dtype=torch.bfloat16,
                quantization_config=DiffusersBitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                ),
            )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )

    pipe.to(f"cuda:{local_rank}")

    if args.cache:
        from cache_dit import (
            BlockAdapter,
            ForwardPattern,
            DBCacheConfig,
            TaylorSeerCalibratorConfig,
        )

        assert isinstance(pipe.dit, LongCatVideoTransformer3DModel)

        # Using Cache-DiT to cache the DiT transformer blocks of LongCat-Video
        cache_dit.enable_cache(
            BlockAdapter(
                transformer=pipe.dit,
                blocks=pipe.dit.blocks,
                forward_pattern=ForwardPattern.Pattern_3,
                check_forward_pattern=False,
                has_separate_cfg=False,
            ),
            cache_config=DBCacheConfig(
                Fn_compute_blocks=args.Fn,
                Bn_compute_blocks=args.Bn,
                max_warmup_steps=args.max_warmup_steps,
                max_cached_steps=args.max_cached_steps,
                max_continuous_cached_steps=args.max_continuous_cached_steps,
                residual_diff_threshold=args.rdt,
                # NOTE: num_inference_steps is required for Transformer-only interface
                num_inference_steps=50 if args.steps is None else args.steps,
            ),
            calibrator_config=(
                TaylorSeerCalibratorConfig(
                    taylorseer_order=args.taylorseer_order,
                )
                if args.taylorseer
                else None
            ),
        )

    global_seed = 42
    seed = global_seed + global_rank

    def run_t2v():
        # t2v (480p)
        output = pipe.generate_t2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480 if args.height is None else args.height,
            width=832 if args.width is None else args.width,
            num_frames=93 if args.frames is None else args.frames,
            num_inference_steps=50 if args.steps is None else args.steps,
            guidance_scale=4.0,
            generator=torch.Generator(device=local_rank).manual_seed(seed),
        )[0]
        return output

    if args.compile:
        pipe.dit = torch.compile(pipe.dit)

        # warmup
        _ = run_t2v()
        torch_gc()

    start = time.time()
    output = run_t2v()
    end = time.time()

    if local_rank == 0:
        cache_dit.summary(pipe.dit)

        time_cost = end - start
        save_path = f"longcat-video.{strify(args, pipe.dit)}"
        if args.quantize:
            save_path += ".bnb4bits" if context_parallel_size < 4 else ".fp8wo"
        save_path += ".mp4"
        print(f"Time cost: {time_cost:.2f}s")
        print(f"Saving video to {save_path}")

        output_tensor = torch.from_numpy(np.array(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        write_video(
            save_path,
            output_tensor,
            fps=15,
            video_codec="libx264",
            options={"crf": f"{18}"},
        )
    del output
    torch_gc()

    if dist.is_initialized():
        dist.destroy_process_group()


def _parse_args():
    DEAULT_CHECKPOINT_DIR = os.environ.get("LONGCAT_VIDEO_DIR", None)
    parser = get_args(parse=False)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--context_parallel_size", type=int, default=None)
    parser.add_argument(
        "--checkpoint_dir", type=str, default=DEAULT_CHECKPOINT_DIR
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
