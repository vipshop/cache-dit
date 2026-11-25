"""Benchmark full MiniMax-H3 video VAE decode on two GPUs."""

import argparse
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLMiniMaxH3
from diffusers import __version__ as diffusers_version

from cache_dit import ParallelismConfig
from cache_dit.distributed.autoencoders import (
  parallelize_autoencoder,
  set_minimax_h3_vae_batch_local_tiles,
)

NUM_RANKS = 2
NUM_TRIALS = 3
SEED = 17
LATENT_SHAPE = (1, 24, 37, 48, 84)
VIDEO_SHAPE = (1, 3, 124, 768, 1344)
DECODER_CALLS = {
  "replicated": (196, 1),
  "parallel": (98, 1),
  "batched": (7, 14),
}


@torch.no_grad()
def _measure_decode(
  vae: AutoencoderKLMiniMaxH3,
  latents: torch.Tensor,
  mode: str,
) -> tuple[list[float], int]:
  device = latents.device
  calls, batch_size = DECODER_CALLS[mode]
  batches: list[int] = []
  # Verify the selected path during warmup to keep the hook out of the timed trials.
  with vae.decoder.register_forward_pre_hook(
      lambda _module, inputs: batches.append(inputs[0].shape[0])):
    video = vae.decode(latents, return_dict=False)[0]
  torch.cuda.synchronize(device)
  assert tuple(video.shape) == VIDEO_SHAPE, f"Unexpected decoded video shape {video.shape}"
  assert batches == [batch_size] * calls, (
    f"{mode} expected {calls} decoder calls at batch size {batch_size}, got {batches}")
  del video

  decode_seconds: list[float] = []
  peak_allocated_bytes = 0
  for _ in range(NUM_TRIALS):
    dist.barrier()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    video = vae.decode(latents, return_dict=False)[0]
    torch.cuda.synchronize(device)
    decode_seconds.append(perf_counter() - started)
    peak_allocated_bytes = max(peak_allocated_bytes, torch.cuda.max_memory_allocated(device))
    del video
  return decode_seconds, peak_allocated_bytes


def main() -> None:
  """Load the VAE and measure the selected mode on both ranks."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--vae-dir", required=True, type=Path, help="Local MiniMax-H3 VAE directory")
  parser.add_argument("--mode", required=True, choices=DECODER_CALLS)
  args = parser.parse_args()
  if os.environ.get("WORLD_SIZE") != str(NUM_RANKS):
    parser.error("Launch with torchrun --standalone --nproc-per-node=2")

  device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
  torch.cuda.set_device(device)
  dist.init_process_group("nccl", device_id=device)
  try:
    vae_dir = args.vae_dir.resolve()
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
      vae_dir,
      torch_dtype=torch.float32,
      local_files_only=True,
    ).to(device)
    vae.enable_tiling()
    if args.mode != "replicated":
      vae = parallelize_autoencoder(vae, ParallelismConfig(tp_size=NUM_RANKS))
      set_minimax_h3_vae_batch_local_tiles(vae, enabled=args.mode == "batched")

    generator = torch.Generator("cpu").manual_seed(SEED)
    latents = torch.randn(LATENT_SHAPE, generator=generator, dtype=torch.float32, device="cpu")
    mean = latents.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = latents.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    latents.mul_(std).add_(mean)

    decode_seconds, peak_allocated_bytes = _measure_decode(vae, latents.to(device), args.mode)
    peak = torch.tensor(peak_allocated_bytes, device=device, dtype=torch.int64)
    dist.reduce(peak, dst=0, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
      calls, batch_size = DECODER_CALLS[args.mode]
      report = {
        "mode": args.mode,
        "vae_dir": str(vae_dir),
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "diffusers": diffusers_version,
        "latent_shape": LATENT_SHAPE,
        "seed": SEED,
        "decoder_calls_per_rank": calls,
        "decoder_batch_size": batch_size,
        "decode_seconds": decode_seconds,
        "median_decode_seconds": median(decode_seconds),
        "peak_allocated_gib": peak.item() / 2 ** 30,
      }
      print(json.dumps(report, indent=2))
  finally:
    dist.destroy_process_group()


if __name__ == "__main__":
  main()
