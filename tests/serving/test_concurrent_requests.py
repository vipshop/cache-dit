import asyncio
import aiohttp
import time


async def send_request(session, url, request_id):
    payload = {
        "prompt": f"test prompt {request_id}",
        "width": 512,
        "height": 512,
        "num_inference_steps": 10,
        "guidance_scale": 3.5,
        "seed": request_id,
        "num_images": 1,
    }

    start_time = time.time()
    async with session.post(url, json=payload) as response:
        elapsed = time.time() - start_time
        result = await response.json() if response.status == 200 else {}
        return {
            "id": request_id,
            "status": response.status,
            "elapsed": elapsed,
            "server_time": result.get("time_cost", 0),
        }


async def main():
    url = "http://localhost:8000/generate"
    num_requests = 3

    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, url, i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)

    total_time = time.time() - start_time

    print(f"Total: {total_time:.2f}s")
    for r in results:
        print(
            f"Request {r['id']}: status={r['status']}, elapsed={r['elapsed']:.2f}s, server={r['server_time']:.2f}s"
        )


if __name__ == "__main__":
    asyncio.run(main())
