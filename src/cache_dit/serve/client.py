"""Simple client for cache-dit serving API."""

import argparse
import base64
import requests
from io import BytesIO
from PIL import Image


def generate_image(
    prompt: str,
    host: str = "localhost",
    port: int = 8000,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: int = None,
    output: str = "output.png",
):
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }

    if seed is not None:
        payload["seed"] = seed

    print(f"Generating image: {prompt}")
    response = requests.post(url, json=payload)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    result = response.json()

    img_data = base64.b64decode(result["images"][0])
    img = Image.open(BytesIO(img_data))
    img.save(output)

    print(f"Image saved to {output}")

    if result.get("stats"):
        print(f"Cache stats: {result['stats']}")
    if result.get("time_cost"):
        print(f"Time cost: {result['time_cost']:.2f}s")

    if result.get("inference_start_time"):
        print(f"Inference start time: {result['inference_start_time']}")
    if result.get("inference_end_time"):
        print(f"Inference end time: {result['inference_end_time']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache-DiT serving client")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output file")

    args = parser.parse_args()

    generate_image(
        prompt=args.prompt,
        host=args.host,
        port=args.port,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        output=args.output,
    )
