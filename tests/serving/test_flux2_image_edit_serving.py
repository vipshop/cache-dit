import os
import requests
import base64
from PIL import Image
from io import BytesIO


def call_api(prompt, image_urls=None, name="test", **kwargs):
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": prompt,
        "width": kwargs.get("width", 1024),
        "height": kwargs.get("height", 1024),
        "num_inference_steps": kwargs.get("num_inference_steps", 50),
        "guidance_scale": kwargs.get("guidance_scale", 4.0),
        "seed": kwargs.get("seed", 42),
    }

    if image_urls:
        payload["image_urls"] = image_urls

    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()

        if "images" not in result or len(result["images"]) == 0:
            return None

        img_data = base64.b64decode(result["images"][0])
        img = Image.open(BytesIO(img_data))

        filename = f"{name}.png"
        img.save(filename)

        print(f"Saved: {filename}")
        return filename

    except Exception as e:
        print(f"Error: {e}")
        return None


def test_single():
    return call_api(
        prompt="Put a birthday hat on the dog in the image",
        image_urls=["https://modelscope.oss-cn-beijing.aliyuncs.com/Dog.png"],
        name="single_edit",
    )


def test_multi():
    return call_api(
        prompt="Realistic style, dog chases frisbee",
        image_urls=[
            "https://modelscope.oss-cn-beijing.aliyuncs.com/Dog.png",
            "https://modelscope.oss-cn-beijing.aliyuncs.com/Frisbee.png",
        ],
        name="multi_edit",
    )


def test_base64():
    image_url = "https://modelscope.oss-cn-beijing.aliyuncs.com/Dog.png"
    response = requests.get(image_url, timeout=30)
    img_base64 = base64.b64encode(response.content).decode("utf-8")

    filename1 = call_api(
        prompt="Put a birthday hat on the dog", image_urls=[img_base64], name="base64_raw"
    )

    data_uri = f"data:image/png;base64,{img_base64}"
    filename2 = call_api(
        prompt="Put a birthday hat on the dog", image_urls=[data_uri], name="base64_uri"
    )

    return filename1, filename2


def test_text():
    return call_api(
        prompt="A beautiful landscape with mountains and lakes",
        name="text_gen",
        num_inference_steps=28,
    )


if __name__ == "__main__":
    test_single()
    test_multi()
    test_base64()
    test_text()
