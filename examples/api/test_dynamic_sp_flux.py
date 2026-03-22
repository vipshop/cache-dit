"""Launch with torchrun; --nproc_per_node must match the YAML (ulysses_size / SP world).

  torchrun --nproc_per_node=2 examples/api/test_dynamic_sp_2gpu_flux.py
  torchrun --nproc_per_node=4 examples/api/test_dynamic_sp_2gpu_flux.py

If --config is omitted, picks examples/configs/dynamic_sp_{N}gpu.yaml when present,
else examples/configs/dynamic_sp.yaml for world_size==8.
"""

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from diffusers import FluxPipeline

import cache_dit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed validation for dynamic SP with a real FLUX model (any SP degree)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get(
            "FLUX_DIR",
            "/project/infattllm/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9/",
        ),
        help="HF model id or local model path.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML with parallelism_config.dynamic_sp. "
        "Default: auto examples/configs/dynamic_sp_{N}gpu.yaml (or dynamic_sp.yaml for N=8).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A small robot reading a book in a cozy library, cinematic lighting.",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=str,
        default="dynamic_sp_flux.png",
        help="Output image path (saved on rank 0 only).",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="For FLUX schnell, guidance_scale=0 is a safe default.",
    )
    return parser.parse_args()


def _resolve_config_path(explicit: Optional[str], world_size: int) -> str:
    if explicit is not None:
        return explicit
    candidates = [f"examples/configs/dynamic_sp_{world_size}gpu.yaml"]
    if world_size == 8:
        candidates.append("examples/configs/dynamic_sp.yaml")
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise RuntimeError(
        f"No default config found for world_size={world_size}. Tried {candidates}. "
        "Pass --config to a YAML whose parallelism matches torchrun --nproc_per_node."
    )


def _init_dist() -> tuple[int, int, int, torch.device]:
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, local_rank, world_size, device


def _attach_inactive_counter(transformer: torch.nn.Module) -> list[int]:
    manager = getattr(transformer, "_dynamic_sp_manager", None)
    if manager is None:
        raise RuntimeError(
            "dynamic_sp manager was not attached. "
            "Please check whether dynamic_sp.enabled=true in the config."
        )

    inactive_counter = [0]
    original_sync = manager.sync_output

    def counted_sync(output, hidden_states, step: int):
        if output is None:
            inactive_counter[0] += 1
        return original_sync(output=output, hidden_states=hidden_states, step=step)

    manager.sync_output = counted_sync
    return inactive_counter


def _all_gather_max_diff(image_np: np.ndarray, device: torch.device) -> float:
    image_np = np.ascontiguousarray(image_np)
    image_tensor = torch.from_numpy(image_np).to(device=device, dtype=torch.float32).contiguous()
    gathered = [torch.empty_like(image_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, image_tensor)

    ref = gathered[0]
    max_diff = 0.0
    for other in gathered[1:]:
        diff = (other - ref).abs().max().item()
        max_diff = max(max_diff, float(diff))
    return max_diff


def main():
    args = parse_args()
    rank, _, world_size, device = _init_dist()
    config_path = _resolve_config_path(args.config, world_size)
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(device)
    configs = cache_dit.load_configs(config_path)
    cache_dit.enable_cache(pipe, **configs)

    inactive_counter = _attach_inactive_counter(pipe.transformer)
    manager = pipe.transformer._dynamic_sp_manager

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    output = pipe(
        args.prompt,
        num_inference_steps=args.steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        generator=generator,
        output_type="np",
    )
    image_np = output.images[0]

    max_diff = _all_gather_max_diff(image_np=image_np, device=device)

    inactive_tensor = torch.tensor([inactive_counter[0]], device=device, dtype=torch.int64)
    dist.all_reduce(inactive_tensor, op=dist.ReduceOp.SUM)
    total_inactive_calls = int(inactive_tensor.item())

    manager_steps = torch.tensor([int(manager.step)], device=device, dtype=torch.int64)
    min_steps = manager_steps.clone()
    max_steps = manager_steps.clone()
    dist.all_reduce(min_steps, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_steps, op=dist.ReduceOp.MAX)
    expects_inactive = any(
        len(entry[2]) < world_size for entry in getattr(manager, "_schedule", [])
    )

    # Validation conditions:
    # 1) manager stepped at least once
    # 2) inactive branch executed on at least one rank
    # 3) final outputs are numerically aligned across both ranks
    ok = True
    reason: Optional[str] = None
    if int(min_steps.item()) <= 0:
        ok = False
        reason = f"manager.step did not increase (min_steps={int(min_steps.item())})."
    elif expects_inactive and total_inactive_calls <= 0:
        ok = False
        reason = "inactive branch was never triggered; dynamic schedule may not be applied."
    elif max_diff > 1e-5:
        ok = False
        reason = f"rank output mismatch too large (max_diff={max_diff})."

    if rank == 0:
        print("=" * 80)
        print("Dynamic SP 2-GPU validation summary")
        print(f"model: {args.model}")
        print(f"config: {args.config}")
        print(f"steps: {args.steps}")
        print(
            f"manager.step range across ranks: [{int(min_steps.item())}, {int(max_steps.item())}]"
        )
        print(f"total inactive sync calls across ranks: {total_inactive_calls}")
        print(f"max output diff across ranks: {max_diff:.8f}")
        print(f"result: {'PASS' if ok else 'FAIL'}")
        if reason is not None:
            print(f"reason: {reason}")
        print("=" * 80)

        if ok:
            pil_image = pipe.numpy_to_pil(image_np[None, ...])[0]
            pil_image.save(args.output)
            print(f"saved image: {args.output}")

    dist.barrier()
    if not ok:
        raise RuntimeError(reason or "Dynamic SP 2-GPU validation failed.")


if __name__ == "__main__":
    main()
