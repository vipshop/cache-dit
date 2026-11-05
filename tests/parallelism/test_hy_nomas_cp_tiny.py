"""Multi-rank parity test for the HY-Nomas context-parallel planner (tiny model, no weights).

HY-Nomas keeps a **persistent** KV cache across segments, so a CP bug does not only change the
current segment: it writes wrong history that every later segment reads.  Two properties are locked
here, in float32 so the tolerance is round-off:

* the video tokens are split on ``double_blocks.0`` and gathered on ``final_layer`` (outside the
  cached block container), so a forward reproduces the single-rank output;
* each rank's collected K rows are **bit-identical** to that rank's head slice of the single-rank K
  (``k[:, :, r * H_local:(r + 1) * H_local]``), which is what the cache reuses next segment.

Usage:
  CUDA_VISIBLE_DEVICES=6,0 python -m pytest tests/parallelism/test_hy_nomas_cp_tiny.py -q
"""

import os

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
_PORT = 29631


def _build():
  from cache_dit.models.pipelines.hy_nomas import NomasTransformer3DModel

  torch.manual_seed(0)
  model = NomasTransformer3DModel(
    hidden_size=_HIDDEN,
    num_attention_heads=_HEADS,
    num_layers=_LAYERS,
    in_channels=_IN_CHANNELS,
    out_channels=_IN_CHANNELS,
    text_states_dim=_TEXT_DIM,
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


def _run(model, inputs, device):
  cache = model.allocate_cache(_CAPACITY)
  with torch.inference_mode():
    out = model(**_to_device(inputs, device), cache=cache)
    _, entries = model.predict(inputs["hidden_states"].to(device),
                               inputs["timestep"].to(device),
                               inputs["encoder_hidden_states"].to(device),
                               inputs["text_mask"].to(device),
                               cache=cache,
                               collect_cache=True)
  return out.float().cpu(), entries[0]["k"].float().cpu()


def _run_rank(rank: int, world: int) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    from cache_dit.distributed import enable_parallelism
    from cache_dit.distributed.config import ParallelismConfig

    inputs = _inputs()
    model = _build().to(device)

    reference_out, reference_k = None, None
    if rank == 0:
      reference_out, reference_k = _run(model, inputs, device)
    torch.distributed.barrier()

    transformer = enable_parallelism(model, ParallelismConfig(ulysses_size=world))
    local_heads = transformer.config.num_attention_heads // world
    for block in transformer.double_blocks:
      assert block.attn_heads == local_heads, f"attn_heads not divided: {block.attn_heads}"

    out, key = _run(transformer, inputs, device)
    # The group is bound lazily from `_cp_config` when the segment context is built.
    assert transformer.parallel.group is not None, "the CP group was never bound"

    if rank == 0:
      relative = (out - reference_out).abs().mean() / reference_out.abs().mean().clamp_min(1e-6)
      print(f"[hy-nomas-cp] rel={relative.item():.8f} shape={tuple(out.shape)}")
      assert out.shape == reference_out.shape, f"cp changed the output shape: {out.shape}"
      assert relative.item() < _REL_TOL, f"cp diverged (rel={relative.item():.8f})"

    # Every rank's K must equal its own head slice of the single-rank K, bit for bit.
    gathered = [None] * world
    torch.distributed.all_gather_object(gathered, key.cpu())
    if rank == 0:
      tokens = reference_k.shape[1]
      for source, shard in enumerate(gathered):
        expected = reference_k[:, :, source * local_heads:(source + 1) * local_heads]
        assert shard.shape[1] == tokens, f"rank {source} stored {shard.shape[1]} rows of {tokens}"
        delta = (shard - expected).abs().max().item()
        print(f"[hy-nomas-cp] rank{source} k max|delta|={delta:.3e}")
        assert delta == 0.0, f"rank {source} K is not bit-identical (max|delta|={delta:.3e})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_hy_nomas_cp_matches_single_rank():
  mp.spawn(_run_rank, args=(2, ), nprocs=2, join=True)
