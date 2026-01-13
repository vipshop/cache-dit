"""Test LTX-2 Image-to-Video model serving.

Server setup (base model):
    CACHE_DIT_LTX2_PIPELINE=i2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        -m cache_dit.serve.serve \
        --model-path Lightricks/LTX-2 \
        --parallel-type ulysses \
        --parallel-vae \
        --cache \
        --ulysses-anything

 Server setup (base model, TP4):
     CACHE_DIT_LTX2_PIPELINE=i2v CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
         -m cache_dit.serve.serve \
         --model-path Lightricks/LTX-2 \
         --parallel-type tp \
         --cache

Server setup (base + LoRA):
    # NOTE: the LoRA weight filename may differ. Common filenames include:
    # - pytorch_lora_weights.safetensors
    # - adapter_model.safetensors
    CACHE_DIT_LTX2_PIPELINE=i2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        -m cache_dit.serve.serve \
        --model-path Lightricks/LTX-2 \
        --lora-path Lightricks/LTX-2-19b-IC-LoRA-Canny-Control \
        --lora-name ltx-2-19b-ic-lora-canny-control.safetensors \
        --parallel-type ulysses \
        --parallel-vae \
        --cache \
        --ulysses-anything

 Server setup (base + LoRA, TP4):
     # NOTE: the LoRA weight filename may differ. Common filenames include:
     # - pytorch_lora_weights.safetensors
     # - adapter_model.safetensors
     CACHE_DIT_LTX2_PIPELINE=i2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
         -m cache_dit.serve.serve \
         --model-path Lightricks/LTX-2 \
         --lora-path Lightricks/LTX-2-19b-IC-LoRA-Canny-Control \
         --lora-name ltx-2-19b-ic-lora-canny-control.safetensors \
         --parallel-type tp \
         --cache

Usage:
    # Base model
    CACHE_DIT_LTX2_MODE=base python -m pytest -q cache-dit/tests/serving/test_ltx2_image2video.py

    # LoRA model
    CACHE_DIT_LTX2_MODE=lora python -m pytest -q cache-dit/tests/serving/test_ltx2_image2video.py
"""

import os
import base64
import requests


def call_api(prompt, image_url, name="ltx2_i2v", **kwargs):
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": prompt,
        "negative_prompt": kwargs.get("negative_prompt", ""),
        "image_urls": [image_url],
        "width": kwargs.get("width", 768),
        "height": kwargs.get("height", 512),
        "num_inference_steps": kwargs.get("num_inference_steps", 40),
        "guidance_scale": kwargs.get("guidance_scale", 4.0),
        "seed": kwargs.get("seed", 1234),
        "num_frames": kwargs.get("num_frames", 121),
        "fps": kwargs.get("fps", 24),
    }

    if "output_format" in kwargs:
        payload["output_format"] = kwargs["output_format"]
    if "output_dir" in kwargs:
        payload["output_dir"] = kwargs["output_dir"]

    response = requests.post(url, json=payload, timeout=1800)
    response.raise_for_status()
    result = response.json()

    assert "video" in result and result["video"] is not None, f"No video in response: keys={list(result.keys())}"

    if payload.get("output_format", "base64") == "path":
        filename = result["video"]
        assert os.path.exists(filename)
        print(f"Saved: {filename}")
        return filename

    video_data = base64.b64decode(result["video"])
    filename = f"{name}.mp4"
    with open(filename, "wb") as f:
        f.write(video_data)
    print(f"Saved: {filename}")
    return filename


def test_ltx2_image2video():
    # Align with upstream diffusers LTX2 example.
    image_url = (
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
        "diffusers/astronaut.jpg"
    )

    mode = os.environ.get("CACHE_DIT_LTX2_MODE", "base").lower()
    if mode not in ("base", "lora"):
        raise ValueError("CACHE_DIT_LTX2_MODE must be 'base' or 'lora'")

    if mode == "base":
        prompt = (
            "An astronaut hatches from a fragile egg on the surface of the Moon, the shell cracking and peeling "
            "apart in gentle low-gravity motion. Fine lunar dust lifts and drifts outward with each movement, "
            "floating in slow arcs before settling back onto the ground. The astronaut pushes free in a deliberate, "
            "weightless motion, small fragments of the egg tumbling and spinning through the air. In the background, "
            "the deep darkness of space subtly shifts as stars glide with the camera's movement, emphasizing vast "
            "depth and scale. The camera performs a smooth, cinematic slow push-in, with natural parallax between the "
            "foreground dust, the astronaut, and the distant starfield. Ultra-realistic detail, physically accurate "
            "low-gravity motion, cinematic lighting, and a breath-taking, movie-like shot."
        )
        negative_prompt = (
            "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, "
            "motion artifacts, fused fingers, bad anatomy, weird hand, ugly, transition, static."
        )
        return call_api(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_url=image_url,
            name="ltx2_base_i2v",
            seed=42,
            width=768,
            height=512,
            num_frames=121,
            fps=24,
            num_inference_steps=40,
            guidance_scale=4.0,
        )

    # LoRA Canny Control variant. The API schema doesn't expose an explicit canny control image field,
    # so we still provide `image_urls` as the conditioning image and a canny-oriented prompt.
    prompt = (
        "Canny edge control style: keep structure of the input image, "
        "turn it into a cinematic animated sequence with strong edges and clean contours."
    )
    negative_prompt = (
        "worst quality, low quality, jpeg artifacts, blurry, deformed, disfigured, "
        "extra limbs, extra fingers, text, watermark"
    )
    return call_api(
        prompt=prompt,
        image_url=image_url,
        name="ltx2_lora_canny_i2v",
        negative_prompt=negative_prompt,
        seed=123,
        fps=24,
    )


if __name__ == "__main__":
    print("Testing LTX-2 Image-to-Video Serving...")
    print(f"CACHE_DIT_LTX2_MODE={os.environ.get('CACHE_DIT_LTX2_MODE', 'base')}")
    test_ltx2_image2video()
    print("Done.")


