import os
import requests
import base64
from PIL import Image
from io import BytesIO


def call_api(prompt, image_url, name="test", **kwargs):
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": prompt,
        "image_urls": [image_url],
        "width": kwargs.get("width", 832),
        "height": kwargs.get("height", 480),
        "num_inference_steps": kwargs.get("num_inference_steps", 50),
        "guidance_scale": kwargs.get("guidance_scale", 3.5),
        "seed": kwargs.get("seed", 1234),
        "num_frames": kwargs.get("num_frames", 49),
        "fps": kwargs.get("fps", 16),
    }

    if "negative_prompt" in kwargs:
        payload["negative_prompt"] = kwargs["negative_prompt"]

    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()

        if "video" not in result or result["video"] is None:
            return None

        video_data = base64.b64decode(result["video"])
        filename = f"{name}.mp4"

        with open(filename, "wb") as f:
            f.write(video_data)

        print(f"Saved: {filename}")
        return filename

    except Exception as e:
        print(f"Error: {e}")
        return None


def test_basic():
    image_url = (
        "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    )
    prompt = "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."

    return call_api(prompt=prompt, image_url=image_url, name="wan_i2v_basic")


def test_with_negative_prompt():
    image_url = (
        "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    )
    prompt = "A white cat on a surfboard at the beach, enjoying the summer vacation"
    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    return call_api(
        prompt=prompt,
        image_url=image_url,
        negative_prompt=negative_prompt,
        name="wan_i2v_negative",
        seed=42,
    )


def test_custom_image():
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    prompt = "A cat walking in a beautiful garden with flowers"

    return call_api(prompt=prompt, image_url=image_url, name="wan_i2v_custom", seed=999)


def test_short_video():
    image_url = (
        "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    )
    prompt = "A cat on a surfboard, gentle waves in the background"

    return call_api(
        prompt=prompt,
        image_url=image_url,
        name="wan_i2v_short",
        num_frames=25,
        num_inference_steps=30,
        seed=777,
    )


def test_with_base64_image():
    image_url = (
        "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
    )
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    data_uri = f"data:image/jpeg;base64,{img_base64}"

    prompt = "A cat enjoying the beach vacation"

    return call_api(prompt=prompt, image_url=data_uri, name="wan_i2v_base64", seed=555)


if __name__ == "__main__":
    print("Testing Wan Image-to-Video Serving...")
    print("\n1. Basic test:")
    test_basic()

    print("\n2. With negative prompt:")
    test_with_negative_prompt()

    print("\n3. Custom image:")
    test_custom_image()

    print("\n4. Short video:")
    test_short_video()

    print("\n5. Base64 encoded image:")
    test_with_base64_image()

    print("\nAll tests completed!")
