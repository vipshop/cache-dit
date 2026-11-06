"""Multi-rank parity test for the HY-Nomas tensor-parallel planner (tiny model, no weights).

The plan shards the Q/K/V projections, the output projections and the MLPs of every double-stream
block, and keeps the modulation projection sharded **with a replicated output** (the block does
``.chunk(6, -1)`` on it and applies the pieces to a full-width activation).  A plan that shards the
modulation activation instead, or that fails to divide ``block.heads`` together with the projection
width, keeps producing tensors of the right shape, so the only reliable check is a numerical
comparison against a single-rank reference.

Run in float32 so the all-reduce reorder stays at round-off level (``rel < 1e-6``); bf16 would only
prove that the plan is not obviously broken.

Usage:
  CUDA_VISIBLE_DEVICES=6,0 python -m pytest tests/parallelism/test_hy_nomas_tp_tiny.py -q
"""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

_HIDDEN = 32
_HEADS = 2
_LAYERS = 2
_TEXT_DIM = 16
_TEXT_TOKENS = 6
_FRAMES = 2
_SIDE = 2
_IN_CHANNELS = 32
_CAPACITY = 64
_REL_TOL = 1e-6


def _free_port() -> int:
  """Bind port 0 and hand the kernel's choice to the spawn.

  A hardcoded rendezvous port collides whenever a previous case's spawned children still hold it,
  which is exactly what turned the full `tests/parallelism` run flaky.
  """
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
    probe.bind(("127.0.0.1", 0))
    return probe.getsockname()[1]


def _build() -> "torch.nn.Module":
  from cache_dit.models.pipelines.hy_nomas import NomasTransformer3DModel

  torch.manual_seed(0)
  model = NomasTransformer3DModel(
    hidden_size=_HIDDEN,
    num_attention_heads=_HEADS,
    num_layers=_LAYERS,
    in_channels=_IN_CHANNELS,
    out_channels=_IN_CHANNELS,
    text_states_dim=_TEXT_DIM,
    # Tiny head_dim = 16, so the three RoPE axes sum to it instead of the release's 128.
    rope_dim_list=(4, 6, 6),
    attention_backend="sdpa",
  ).eval().requires_grad_(False)
  with torch.no_grad():
    for name, param in model.named_parameters():
      if param.ndim == 1 and "norm" in name and name.endswith("weight"):
        param.fill_(1)
      else:
        param.normal_(0, 0.02)
  return model


def _inputs(seed: int = 1):
  generator = torch.Generator().manual_seed(seed)
  return {
    "hidden_states": torch.randn(1, _IN_CHANNELS, _FRAMES, _SIDE, _SIDE, generator=generator),
    "encoder_hidden_states": torch.randn(1, _TEXT_TOKENS, _TEXT_DIM, generator=generator),
    "timestep": torch.tensor([500.0]),
    "text_mask": (torch.arange(_TEXT_TOKENS)[None] < _TEXT_TOKENS - 1),
  }


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _forward(model, inputs, device):
  cache = model.allocate_cache(_CAPACITY)
  with torch.inference_mode():
    out = model(**_to_device(inputs, device), cache=cache)
  return out.float().cpu()


def _run_rank(rank: int, world: int, port: int) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(port)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    from cache_dit.distributed.config import ParallelismConfig
    from cache_dit.distributed.transformers.dispatch import (
      _parallelize_transformer_tensor, )

    inputs = _inputs()
    model = _build().to(device)

    reference = None
    if rank == 0:
      reference = _forward(model, inputs, device)
    torch.distributed.barrier()

    transformer = _parallelize_transformer_tensor(model, ParallelismConfig(tp_size=world))
    # Guard against a no-op parallelization: an unsharded transformer would also "match".
    from torch.distributed._tensor import DTensor

    block0 = transformer.double_blocks[0]
    assert isinstance(block0.img_attn_q.weight, DTensor), "the Q projection was not sharded"
    assert isinstance(block0.img_mlp.fc1.weight, DTensor), "the MLP was not sharded"
    local_heads = transformer.config.num_attention_heads // world
    for block in transformer.double_blocks:
      assert block.heads == local_heads, f"heads metadata not divided: {block.heads}"
      assert block.attn_heads == local_heads, f"attn_heads not divided: {block.attn_heads}"
    assert transformer.local_attention_heads == local_heads

    out = _forward(transformer, inputs, device)
    if rank == 0:
      relative = (out - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)
      print(f"[hy-nomas-tp] rel={relative.item():.8f} shape={tuple(out.shape)}")
      assert out.shape == reference.shape, f"tp changed the output shape: {out.shape}"
      assert relative.item() < _REL_TOL, f"tp diverged (rel={relative.item():.8f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_hy_nomas_tp_matches_single_rank():
  mp.spawn(_run_rank, args=(2, _free_port()), nprocs=2, join=True)
