import os
import time
import torch

from PIL import Image
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from utils import GiB, get_args
import cache_dit


args = get_args()
print(args)

pipe = QwenImageEditPipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_EDIT_DIR",
        "Qwen/Qwen-Image-Edit",
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
)

if args.cache:
    cache_dit.enable_cache(
        pipe,
        # Cache context kwargs
        do_separate_cfg=True,
        enable_taylorseer=True,
        enable_encoder_taylorseer=True,
        taylorseer_order=4,
        residual_diff_threshold=0.12,
    )


if torch.cuda.device_count() <= 1:
    # Enable memory savings
    pipe.enable_model_cpu_offload()


image = Image.open("./data/bear.png").convert("RGB")
prompt = "Only change the bear's color to purple"

if args.compile:
    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 8192
    pipe.transformer.compile_repeated_blocks(mode="default")

    # Warmup
    image = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=" ",
        generator=torch.Generator(device="cpu").manual_seed(0),
        true_cfg_scale=4.0,
        num_inference_steps=50,
    ).images[0]

start = time.time()

image = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=" ",
    generator=torch.Generator(device="cpu").manual_seed(0),
    true_cfg_scale=4.0,
    num_inference_steps=50,
).images[0]

end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = (
    f"qwen-image-edit.C{int(args.compile)}_{cache_dit.strify(stats)}.png"
)
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
