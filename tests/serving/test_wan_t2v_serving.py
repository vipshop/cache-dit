import os
import requests
import base64


def call_api(prompt, name="test", **kwargs):
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": prompt,
        "width": kwargs.get("width", 832),
        "height": kwargs.get("height", 480),
        "num_inference_steps": kwargs.get("num_inference_steps", 30),
        "guidance_scale": kwargs.get("guidance_scale", 5.0),
        "seed": kwargs.get("seed", 1234),
        "num_frames": kwargs.get("num_frames", 49),
        "fps": kwargs.get("fps", 16),
    }

    if "output_format" in kwargs:
        payload["output_format"] = kwargs["output_format"]
    if "output_dir" in kwargs:
        payload["output_dir"] = kwargs["output_dir"]

    if "negative_prompt" in kwargs:
        payload["negative_prompt"] = kwargs["negative_prompt"]

    try:
        response = requests.post(url, json=payload, timeout=600)
        response.raise_for_status()
        result = response.json()

        if "video" not in result or result["video"] is None:
            return None

        if payload.get("output_format", "base64") == "path":
            filename = result["video"]
            assert os.path.exists(filename)
            print(f"Saved: {filename}")
            return filename
        else:
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
    return call_api(prompt="A cat walks on the grass, realistic", name="wan_t2v_basic")


def test_custom_prompt():
    return call_api(
        prompt="A beautiful sunset over the ocean with waves crashing on the shore",
        name="wan_t2v_sunset",
        seed=0,
    )


def test_path_output():
    return call_api(
        prompt="A cat walks on the grass, realistic",
        name="wan_t2v_path",
        output_format="path",
        output_dir="outputs_test",
    )


def test_with_negative_prompt():
    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, "
        "style, works, paintings, images, static, overall gray, worst quality, "
        "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background, "
        "three legs, many people in the background, walking backwards"
    )

    return call_api(
        prompt="A dog running in a park, high quality, realistic",
        negative_prompt=negative_prompt,
        name="wan_t2v_negative",
        seed=999,
    )


def test_short_video():
    return call_api(
        prompt="A bird flying in the sky",
        name="wan_t2v_short",
        num_frames=25,
        num_inference_steps=20,
        seed=777,
    )


def test_different_resolution():
    return call_api(
        prompt="A car driving on a highway",
        name="wan_t2v_resolution",
        width=1024,
        height=576,
        seed=555,
    )


if __name__ == "__main__":
    test_basic()
    test_custom_prompt()
    test_path_output()
    test_with_negative_prompt()
    test_short_video()
    test_different_resolution()
