import requests
import base64
from PIL import Image
from io import BytesIO


def test_text_to_image():
    response = requests.post(
        "http://localhost:8000/generate",
        json={
            "prompt": "A beautiful sunset over the ocean",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,
        },
    )

    img_data = base64.b64decode(response.json()["images"][0])
    Image.open(BytesIO(img_data)).save("output.png")
    print("Saved: output.png")


if __name__ == "__main__":
    print("Testing Text-to-Image Serving API...")
    test_text_to_image()
