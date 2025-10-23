import os
import sys

sys.path.append("..")

import time
import torch
import torch.distributed as dist
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    ContextParallelConfig,
)

from utils import maybe_init_distributed, maybe_destroy_distributed


rank, device = maybe_init_distributed()

pipe = QwenImagePipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_DIR",
        "Qwen/Qwen-Image",
    ),
    torch_dtype=torch.bfloat16,
)

# NOTE: Enable cpu offload before enabling parallelism will
# raise shape error after first pipe call, so we enable it after.
# It seems a bug of diffusers that cpu offload is not fully
# compatible with context parallelism, visa versa.
# pipe.enable_model_cpu_offload(device=device)

assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
# pipe.transformer.set_attention_backend("flash")
pipe.transformer.set_attention_backend("_native_cudnn")
pipe.transformer.enable_parallelism(
    config=ContextParallelConfig(ulysses_degree=dist.get_world_size())
)

# NOTE: Enable cpu offload after enabling parallelism
pipe.enable_model_cpu_offload(device=device)

# assert isinstance(pipe.vae, AutoencoderKLQwenImage)
# pipe.vae.enable_tiling()

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
}

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

# using an empty string if you do not have specific concept to remove
negative_prompt = " "

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe():
    # do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    image = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]

    return image


# warmup
_ = run_pipe()

start = time.time()
image = run_pipe()
end = time.time()


if rank == 0:
    time_cost = end - start
    save_path = f"qwen-image.cp{dist.get_world_size()}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
