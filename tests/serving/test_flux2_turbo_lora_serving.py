"""Test FLUX.2 Turbo LoRA model serving.

Server setup:
    CUDA_VISIBLE_DEVICES=0 cache-dit-serve \
        --model-path black-forest-labs/FLUX.2-dev \
        --lora-path fal/FLUX.2-dev-Turbo \
        --lora-name flux.2-turbo-lora.safetensors \
        --dtype bfloat16 \
        --cache

This test calls /generate with a custom sigma schedule (TURBO_SIGMAS) for 8-step turbo inference.

Reference LoRA: https://huggingface.co/fal/FLUX.2-dev-Turbo
Base model: https://huggingface.co/black-forest-labs/FLUX.2-dev
"""

import os
import requests
import base64
from PIL import Image
from io import BytesIO


# Pre-shifted custom sigmas for 8-step turbo inference
TURBO_SIGMAS = [1.0, 0.6509, 0.4374, 0.2932, 0.1893, 0.1108, 0.0495, 0.00031]


def call_api(prompt, name="flux2_turbo", **kwargs):
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": prompt,
        "width": kwargs.get("width", 1024),
        "height": kwargs.get("height", 1024),
        "num_inference_steps": kwargs.get("num_inference_steps", 8),
        "guidance_scale": kwargs.get("guidance_scale", 2.5),
        "sigmas": kwargs.get("sigmas", TURBO_SIGMAS),
        "seed": kwargs.get("seed", 42),
        "num_images": kwargs.get("num_images", 1),
    }

    if "output_format" in kwargs:
        payload["output_format"] = kwargs["output_format"]
    if "output_dir" in kwargs:
        payload["output_dir"] = kwargs["output_dir"]

    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    result = response.json()

    assert "images" in result and result["images"], "No images in response"

    if payload.get("output_format", "base64") == "path":
        filename = result["images"][0]
        assert os.path.exists(filename)
        img = Image.open(filename)
        print(f"Saved: {filename} ({img.size[0]}x{img.size[1]})")
        return filename

    img_data = base64.b64decode(result["images"][0])
    img = Image.open(BytesIO(img_data))

    filename = f"{name}.png"
    img.save(filename)
    print(f"Saved: {filename} ({img.size[0]}x{img.size[1]})")
    return filename


def test_flux2_turbo_lora():
    prompt = (
        "Industrial product shot of a chrome turbocharger with glowing hot exhaust manifold, "
        "engraved text 'FLUX.2 [dev] Turbo by fal' on the compressor housing and 'fal' on the turbine wheel, "
        "gradient heat glow from orange to electric blue , studio lighting with dramatic shadows, "
        "shallow depth of field, engineering blueprint pattern in background."
    )

    return call_api(
        prompt=prompt,
        name="flux2_turbo_lora",
        num_inference_steps=8,
        guidance_scale=2.5,
        sigmas=TURBO_SIGMAS,
        width=1024,
        height=1024,
        seed=42,
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Testing FLUX.2 Turbo LoRA Model Serving")
    print("=" * 80)
    test_flux2_turbo_lora()
    print("=" * 80)
    print("Done")
    print("=" * 80)
