"""Test LTX-2 Text-to-Video model serving.

Server setup (base model):
    CACHE_DIT_LTX2_PIPELINE=t2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        -m cache_dit.serve.serve \
        --model-path Lightricks/LTX-2 \
        --parallel-type ulysses \
        --parallel-text-encoder \
        --parallel-vae \
        --cache \
        --ulysses-anything

 Server setup (base model, TP4):
     CACHE_DIT_LTX2_PIPELINE=t2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
         -m cache_dit.serve.serve \
         --model-path Lightricks/LTX-2 \
         --parallel-type tp \
         --cache

Server setup (base + LoRA):
    # NOTE: the LoRA weight filename may differ. Common filenames include:
    # - pytorch_lora_weights.safetensors
    # - adapter_model.safetensors
    CACHE_DIT_LTX2_PIPELINE=t2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
        -m cache_dit.serve.serve \
        --model-path Lightricks/LTX-2 \
        --lora-path Lightricks/LTX-2-19b-IC-LoRA-Canny-Control \
        --lora-name ltx-2-19b-ic-lora-canny-control.safetensors \
        --parallel-type ulysses \
        --parallel-text-encoder \
        --parallel-vae \
        --cache \
        --ulysses-anything

 Server setup (base + LoRA, TP4):
     # NOTE: the LoRA weight filename may differ. Common filenames include:
     # - pytorch_lora_weights.safetensors
     # - adapter_model.safetensors
     CACHE_DIT_LTX2_PIPELINE=t2v CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
         -m cache_dit.serve.serve \
         --model-path Lightricks/LTX-2 \
         --lora-path Lightricks/LTX-2-19b-IC-LoRA-Canny-Control \
         --lora-name ltx-2-19b-ic-lora-canny-control.safetensors \
         --parallel-type tp \
         --cache

Usage:
    # Base model
    CACHE_DIT_LTX2_MODE=base python -m pytest -q cache-dit/tests/serving/test_ltx2_text2video.py

    # LoRA model
    CACHE_DIT_LTX2_MODE=lora python -m pytest -q cache-dit/tests/serving/test_ltx2_text2video.py
"""

import os
import base64
import requests


def call_api(prompt, name="ltx2_t2v", **kwargs):
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": prompt,
        "negative_prompt": kwargs.get("negative_prompt", ""),
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

    assert (
        "video" in result and result["video"] is not None
    ), f"No video in response: keys={list(result.keys())}"

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


def test_ltx2_text2video():
    mode = os.environ.get("CACHE_DIT_LTX2_MODE", "base").lower()
    if mode not in ("base", "lora"):
        raise ValueError("CACHE_DIT_LTX2_MODE must be 'base' or 'lora'")

    if mode == "base":
        prompt = (
            "A cinematic tracking shot through a neon-lit rainy cyberpunk street at night. "
            "Reflections shimmer on wet asphalt, holographic signs flicker, and steam rises from vents. "
            "A sleek motorbike glides past the camera in slow motion, droplets scattering in the air. "
            "Smooth camera motion, natural parallax, ultra-realistic detail, cinematic lighting, film look."
        )
        negative_prompt = (
            "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion smear, "
            "motion artifacts, bad anatomy, ugly, transition, static, text, watermark."
        )
        return call_api(
            prompt=prompt,
            negative_prompt=negative_prompt,
            name="ltx2_base_t2v",
            seed=42,
            width=768,
            height=512,
            num_frames=121,
            fps=24,
            num_inference_steps=40,
            guidance_scale=4.0,
        )

    # LoRA mode: for validation that serving + LoRA wiring works end-to-end.
    # The same text-only request schema applies.
    prompt = (
        "Cinematic animated sequence, strong edges and clean contours, high contrast lighting, "
        "a robot walks through a foggy corridor, smooth camera dolly in."
    )
    negative_prompt = (
        "worst quality, low quality, jpeg artifacts, blurry, deformed, disfigured, "
        "extra limbs, extra fingers, text, watermark"
    )
    return call_api(
        prompt=prompt,
        negative_prompt=negative_prompt,
        name="ltx2_lora_t2v",
        seed=123,
        fps=24,
    )


if __name__ == "__main__":
    print("Testing LTX-2 Text-to-Video Serving...")
    print(f"CACHE_DIT_LTX2_MODE={os.environ.get('CACHE_DIT_LTX2_MODE', 'base')}")
    test_ltx2_text2video()
    print("Done.")
