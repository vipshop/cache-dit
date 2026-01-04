import os
import requests
import pytest


def _call_generate_api(**overrides):
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": overrides.get("prompt", "timestamp test prompt"),
        "width": overrides.get("width", 1024),
        "height": overrides.get("height", 1024),
        "num_inference_steps": overrides.get("num_inference_steps", 8),
        "guidance_scale": overrides.get("guidance_scale", 1.0),
        "seed": overrides.get("seed", 42),
    }

    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()


def test_generate_returns_inference_timestamps():
    data = _call_generate_api()

    assert "inference_start_time" in data
    assert "inference_end_time" in data
    assert "time_cost" in data

    start = data["inference_start_time"]
    end = data["inference_end_time"]
    time_cost = data["time_cost"]

    print("start", start)
    print("end", end)
    print("time_cost", time_cost)


if __name__ == "__main__":
    test_generate_returns_inference_timestamps()
