"""Test Qwen-Image-Lightning LoRA model serving.

This test demonstrates how to use cache-dit serving with LoRA models.
Qwen-Image-Lightning is a distilled model that generates high-quality images in 4 or 8 steps.

Server setup:
    cache-dit-serve \
        --model-path Qwen/Qwen-Image \
        --lora-path lightx2v/Qwen-Image-Lightning \
        --lora-name Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors \
        --cache

For 4-step model:
    cache-dit-serve \
        --model-path Qwen/Qwen-Image \
        --lora-path lightx2v/Qwen-Image-Lightning \
        --lora-name Qwen-Image-Lightning-4steps-V1.1-bf16.safetensors \
        --cache

Reference: https://huggingface.co/lightx2v/Qwen-Image-Lightning
"""

import os
import requests
import base64
from PIL import Image
from io import BytesIO


def call_api(prompt, name="test", **kwargs):
    """Call the serving API to generate an image."""
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": prompt,
        "width": kwargs.get("width", 1024),
        "height": kwargs.get("height", 1024),
        "num_inference_steps": kwargs.get("num_inference_steps", 8),
        "guidance_scale": kwargs.get("guidance_scale", 1.0),
        "seed": kwargs.get("seed", 42),
        "num_images": kwargs.get("num_images", 1),
    }

    if "include_stats" in kwargs:
        payload["include_stats"] = kwargs["include_stats"]

    if "negative_prompt" in kwargs:
        payload["negative_prompt"] = kwargs["negative_prompt"]

    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()

    if "images" not in result or not result["images"]:
        print("No images in response")
        return None

    # Save all generated images
    filenames = []
    for idx, img_base64 in enumerate(result["images"]):
        img_data = base64.b64decode(img_base64)
        image = Image.open(BytesIO(img_data))

        if kwargs.get("num_images", 1) > 1:
            filename = f"{name}_{idx}.png"
        else:
            filename = f"{name}.png"

        image.save(filename)
        print(f"Saved: {filename} ({image.size})")
        filenames.append(filename)

    if "stats" in result and result["stats"]:
        print(f"Stats: {result['stats']}")
    if "time_cost" in result:
        print(f"Time cost: {result['time_cost']:.2f}s")

    return filenames


def test_basic_8steps():
    """Test basic image generation with 8 steps (Lightning-8steps model)."""
    return call_api(
        prompt="A beautiful landscape with mountains and a lake, high quality",
        name="qwen_lightning_8steps",
        num_inference_steps=8,
        guidance_scale=1.0,
        seed=42,
    )


def test_include_stats():
    """Test include_stats parameter returns stats."""
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))

    model_info_resp = requests.get(f"http://{host}:{port}/get_model_info", timeout=30)
    model_info_resp.raise_for_status()
    enable_cache = bool(model_info_resp.json().get("enable_cache", False))

    result = requests.post(
        f"http://{host}:{port}/generate",
        json={
            "prompt": "A cute puppy playing in the garden",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 8,
            "guidance_scale": 1.0,
            "seed": 456,
            "num_images": 1,
            "include_stats": True,
        },
        timeout=300,
    )
    result.raise_for_status()
    data = result.json()
    if enable_cache:
        assert "stats" in data


def test_basic_4steps():
    """Test basic image generation with 4 steps (Lightning-4steps model)."""
    return call_api(
        prompt="A beautiful landscape with mountains and a lake, high quality",
        name="qwen_lightning_4steps",
        num_inference_steps=4,
        guidance_scale=1.0,
        seed=42,
    )


def test_different_resolution():
    """Test different image resolution (1536x1024)."""
    return call_api(
        prompt="A wide panoramic view of a mountain range at dawn",
        name="qwen_lightning_landscape",
        width=1536,
        height=1024,
        num_inference_steps=8,
        guidance_scale=1.0,
        seed=123,
    )


def test_batch_generation():
    """Test generating multiple images in one request."""
    return call_api(
        prompt="A cute puppy playing in the garden",
        name="qwen_lightning_batch",
        num_inference_steps=8,
        guidance_scale=1.0,
        num_images=4,
        seed=456,
    )


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Qwen-Image-Lightning LoRA Model Serving")
    print("=" * 80)

    # Run tests
    print("\n[1/4] Testing basic 8-step generation...")
    test_basic_8steps()

    print("\n[2/4] Testing basic 4-step generation...")
    test_basic_4steps()

    print("\n[3/4] Testing different resolution (1536x1024)...")
    test_different_resolution()

    print("\n[4/4] Testing batch generation (4 images)...")
    test_batch_generation()

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
