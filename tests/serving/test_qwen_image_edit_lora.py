"""Qwen-Image-Edit + LoRA serving test.

Server (single GPU):
    CUDA_VISIBLE_DEVICES=0 python -m cache_dit.serve.serve \
      --model-path Qwen/Qwen-Image-Edit-2511 \
      --lora-path /home/lmsys/bbuf/qwen-image-lora3 \
      --lora-name lora3-diffusers.safetensors \
      --cache

Server (2 GPUs, ulysses2):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
      -m cache_dit.serve.serve \
      --model-path Qwen/Qwen-Image-Edit-2511 \
      --lora-path /home/lmsys/bbuf/qwen-image-lora3 \
      --lora-name lora3-diffusers.safetensors \
      --parallel-type ulysses \
      --parallel-text-encoder \
      --cache \
      --ulysses-anything

Run client test:
    CACHE_DIT_HOST=localhost CACHE_DIT_PORT=8000 python -m pytest -q \
      cache-dit/tests/serving/test_qwen_image_edit_lora.py
"""

import base64
import os
from io import BytesIO

import requests
from PIL import Image


def call_api(prompt, image_paths, name="qwen_image_edit_lora", **kwargs):
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": prompt,
        "width": kwargs["width"],
        "height": kwargs["height"],
        "num_inference_steps": kwargs.get("num_inference_steps", 30),
        "guidance_scale": kwargs.get("guidance_scale", 4.0),
        "seed": kwargs.get("seed", 1),
        "num_images": kwargs.get("num_images", 1),
        "image_urls": image_paths,
    }

    if "output_format" in kwargs:
        payload["output_format"] = kwargs["output_format"]
    if "output_dir" in kwargs:
        payload["output_dir"] = kwargs["output_dir"]

    response = requests.post(url, json=payload, timeout=1800)
    response.raise_for_status()
    result = response.json()

    assert (
        "images" in result and result["images"]
    ), f"No images in response: keys={list(result.keys())}"

    if payload.get("output_format", "base64") == "path":
        filename = result["images"][0]
        assert os.path.exists(filename)
        img = Image.open(filename)
        print(f"Saved: {filename} ({img.size[0]}x{img.size[1]})")
        return filename

    img_data = base64.b64decode(result["images"][0])
    img = Image.open(BytesIO(img_data)).convert("RGB")
    filename = f"{name}.png"
    img.save(filename)
    print(f"Saved: {filename} ({img.size[0]}x{img.size[1]})")
    return filename


def test_qwen_image_edit_lora():
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    image_path_0 = os.path.join(images_dir, "input_0.png")
    image_path_1 = os.path.join(images_dir, "input_1.png")

    img0 = Image.open(image_path_0).convert("RGB")
    width, height = img0.size

    prompt = (
        "The first image is the original image, and the second image is erasing the corresponding area, "
        "Please use the following instructions to perform image repair. "
        "Using the provided image, first erase all texts (no language restrictions, titles, slogan, date, number, "
        "logo text) from the image, erase all texts from the image, and finally, keep everything else unchanged."
    )

    filename = call_api(
        prompt=prompt,
        image_paths=[image_path_0, image_path_1],
        name="qwen_image_edit_lora",
        seed=1,
        num_inference_steps=30,
        guidance_scale=4.0,
        width=width,
        height=height,
    )

    out_img = Image.open(filename)
    assert out_img.size == (width, height)
    return filename


if __name__ == "__main__":
    test_qwen_image_edit_lora()
