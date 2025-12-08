"""
Test FLUX.2 image editing via serving API
"""

import os
import requests
import base64
import pytest
from PIL import Image
from io import BytesIO


def test_flux2_single_image_edit():
    """Test single image editing via serving API"""
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": "Put a birthday hat on the dog in the image",
        "image_urls": ["https://modelscope.oss-cn-beijing.aliyuncs.com/Dog.png"],
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "seed": 42,
    }

    print(f"Testing image editing with prompt: {payload['prompt']}")
    print(f"Input images: {payload['image_urls']}")

    try:
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code != 200:
            pytest.skip(f"Server not available or error: {response.status_code}")
        
        result = response.json()
        
        # Verify response structure
        assert "images" in result
        assert len(result["images"]) > 0
        assert "time_cost" in result
        
        # Decode and verify image
        img_data = base64.b64decode(result["images"][0])
        img = Image.open(BytesIO(img_data))
        
        assert img.size[0] > 0
        assert img.size[1] > 0
        
        # Save output for inspection
        output_path = "test_dog_with_hat.png"
        img.save(output_path)
        
        print(f"Edited image saved to {output_path}")
        print(f"Time cost: {result['time_cost']:.2f}s")
        
        if result.get("stats"):
            print(f"Cache stats: {result['stats']}")
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
            
    except requests.exceptions.ConnectionError:
        pytest.skip("Serving server not available")
    except requests.exceptions.Timeout:
        pytest.skip("Request timeout")


def test_flux2_multiple_image_edit():
    """Test multiple image composition editing via serving API"""
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": "Realistic style, generate an image where the dog from the first image chases the frisbee from the second image",
        "image_urls": [
            "https://modelscope.oss-cn-beijing.aliyuncs.com/Dog.png",
            "https://modelscope.oss-cn-beijing.aliyuncs.com/Frisbee.png"
        ],
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "seed": 42,
    }

    print(f"Testing multi-image editing with prompt: {payload['prompt']}")
    print(f"Input images: {payload['image_urls']}")

    try:
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code != 200:
            pytest.skip(f"Server not available or error: {response.status_code}")
        
        result = response.json()
        
        # Verify response structure
        assert "images" in result
        assert len(result["images"]) > 0
        
        # Decode and verify image
        img_data = base64.b64decode(result["images"][0])
        img = Image.open(BytesIO(img_data))
        
        assert img.size[0] > 0
        assert img.size[1] > 0
        
        # Save output for inspection
        output_path = "test_dog_with_frisbee.png"
        img.save(output_path)
        
        print(f"Edited image saved to {output_path}")
        print(f"Time cost: {result['time_cost']:.2f}s")
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
            
    except requests.exceptions.ConnectionError:
        pytest.skip("Serving server not available")
    except requests.exceptions.Timeout:
        pytest.skip("Request timeout")


def test_flux2_base64_image_edit():
    """Test image editing with base64 encoded image"""
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    # Download an image and convert to base64
    image_url = "https://modelscope.oss-cn-beijing.aliyuncs.com/Dog.png"
    
    try:
        print(f"Downloading test image from {image_url}")
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Convert to base64
        img_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # Test 1: Raw base64 string
        print("\n--- Test 1: Raw base64 string ---")
        payload = {
            "prompt": "Put a birthday hat on the dog in the image",
            "image_urls": [img_base64],
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 50,
            "guidance_scale": 4.0,
            "seed": 42,
        }

        print(f"Testing image editing with raw base64 input (length: {len(img_base64)})")
        
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code != 200:
            pytest.skip(f"Server not available or error: {response.status_code}")
        
        result = response.json()
        
        # Verify response structure
        assert "images" in result
        assert len(result["images"]) > 0
        assert "time_cost" in result
        
        # Decode and verify image
        img_data = base64.b64decode(result["images"][0])
        img = Image.open(BytesIO(img_data))
        
        assert img.size[0] > 0
        assert img.size[1] > 0
        
        # Save output for inspection
        output_path = "test_dog_base64_raw.png"
        img.save(output_path)
        
        print(f"Edited image saved to {output_path}")
        print(f"Time cost: {result['time_cost']:.2f}s")
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Test 2: Data URI format
        print("\n--- Test 2: Data URI format ---")
        data_uri = f"data:image/png;base64,{img_base64}"
        payload["image_urls"] = [data_uri]
        
        print(f"Testing image editing with data URI input (length: {len(data_uri)})")
        
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code != 200:
            pytest.skip(f"Server not available or error: {response.status_code}")
        
        result = response.json()
        
        # Verify response structure
        assert "images" in result
        assert len(result["images"]) > 0
        
        # Decode and verify image
        img_data = base64.b64decode(result["images"][0])
        img = Image.open(BytesIO(img_data))
        
        assert img.size[0] > 0
        assert img.size[1] > 0
        
        # Save output for inspection
        output_path = "test_dog_base64_data_uri.png"
        img.save(output_path)
        
        print(f"Edited image saved to {output_path}")
        print(f"Time cost: {result['time_cost']:.2f}s")
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
            
    except requests.exceptions.ConnectionError:
        pytest.skip("Serving server not available")
    except requests.exceptions.Timeout:
        pytest.skip("Request timeout")
    except Exception as e:
        pytest.skip(f"Failed to download test image: {e}")


def test_flux2_text_to_image_still_works():
    """Test that text-to-image still works without image_urls"""
    host = os.environ.get("CACHE_DIT_HOST", "localhost")
    port = int(os.environ.get("CACHE_DIT_PORT", 8000))
    url = f"http://{host}:{port}/generate"

    payload = {
        "prompt": "A beautiful landscape with mountains and lakes",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 28,
        "guidance_scale": 4.0,
        "seed": 42,
    }

    print(f"Testing text-to-image mode: {payload['prompt']}")

    try:
        response = requests.post(url, json=payload, timeout=300)
        
        if response.status_code != 200:
            pytest.skip(f"Server not available or error: {response.status_code}")
        
        result = response.json()
        
        # Verify response structure
        assert "images" in result
        assert len(result["images"]) > 0
        
        # Decode and verify image
        img_data = base64.b64decode(result["images"][0])
        img = Image.open(BytesIO(img_data))
        
        assert img.size[0] > 0
        assert img.size[1] > 0
        
        print(f"Text-to-image works correctly")
        print(f"Time cost: {result['time_cost']:.2f}s")
            
    except requests.exceptions.ConnectionError:
        pytest.skip("Serving server not available")
    except requests.exceptions.Timeout:
        pytest.skip("Request timeout")


if __name__ == "__main__":
    print("=== Testing FLUX.2 Image Editing Serving API ===\n")
    
    print("Test 1: Single Image Editing (URL)")
    test_flux2_single_image_edit()
    
    print("\nTest 2: Multiple Image Editing (URLs)")
    test_flux2_multiple_image_edit()
    
    print("\nTest 3: Base64 Image Editing")
    test_flux2_base64_image_edit()
    
    print("\nTest 4: Text-to-Image (backward compatibility)")
    test_flux2_text_to_image_still_works()
    
    print("\n=== All tests completed ===")

