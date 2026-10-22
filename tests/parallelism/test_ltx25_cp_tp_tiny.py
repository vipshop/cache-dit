"""Multi-rank parity test for the LTX-2.5 CP and TP paths (tiny model, no weights needed).

LTX-2.5 blocks use `LTX2PerturbedAttnProcessor` (a standalone class, not a subclass of
`LTX2AudioVideoAttnProcessor`) and per-head gates. If the CP planner patches only the former, or TP
shards `heads` without sharding `to_gate_logits`, the run still completes but produces wrong
numbers -- which is exactly what this test catches: rank 0 builds a cp-free reference, both ranks
then run the same weights under Ulysses CP / TP, and the gathered output must match.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_ltx25_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

try:
  from diffusers.models.transformers.transformer_ltx2 import LTX2PerturbedAttnProcessor  # noqa: F401
except ImportError:
  pytest.skip("diffusers without LTX-2.5 support", allow_module_level=True)

_VIDEO_TOKENS = 16  # 2 frames x 2 x 4
_AUDIO_TOKENS = 8
_INNER_DIM = 32
_REL_TOL = 2e-2  # bf16 all-reduce order noise on a 2-layer random model
_PORT = 29531


def _build(device):
  from diffusers import LTX2VideoTransformer3DModel

  torch.manual_seed(0)
  transformer = LTX2VideoTransformer3DModel(
    in_channels=8,
    out_channels=8,
    num_attention_heads=4,
    attention_head_dim=8,
    cross_attention_dim=_INNER_DIM,
    vae_scale_factors=(1, 1, 1),
    pos_embed_max_pos=16,
    base_height=8,
    base_width=8,
    gated_attn=True,
    cross_attn_mod=True,
    audio_in_channels=8,
    audio_out_channels=8,
    audio_num_attention_heads=4,
    audio_attention_head_dim=8,
    audio_cross_attention_dim=_INNER_DIM,
    audio_scale_factor=1,
    audio_pos_embed_max_pos=16,
    audio_gated_attn=True,
    audio_cross_attn_mod=True,
    num_layers=2,
    caption_channels=_INNER_DIM,
    rope_type="split",
    use_prompt_embeddings=False,
    perturbed_attn=True,
    ff_bias=False,
    audio_ff_bias=False,
    use_prompt_adaln_single=True,
    use_keyframes_abs_pos_embedding=True,
  )
  torch.nn.init.normal_(transformer.keyframes_abs_pos_embedding, std=0.02)
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs():
  generator = torch.Generator().manual_seed(3)
  inputs = {
    "hidden_states": torch.randn(1, _VIDEO_TOKENS, 8, generator=generator),
    "audio_hidden_states": torch.randn(1, _AUDIO_TOKENS, 8, generator=generator),
    "encoder_hidden_states": torch.randn(1, 4, _INNER_DIM, generator=generator),
    "audio_encoder_hidden_states": torch.randn(1, 4, _INNER_DIM, generator=generator),
    "timestep": torch.full((1, _VIDEO_TOKENS), 750.0),
    "audio_timestep": torch.full((1, ), 750.0),
    "sigma": torch.full((1, ), 0.75),
    "audio_sigma": torch.full((1, ), 0.25),
    "fps": 24.0,
    "audio_num_frames": _AUDIO_TOKENS,
    "use_cross_timestep": True,
    # The DFR pipeline always passes a keyframe mask; it is applied after `hidden_states` was
    # already sharded, so CP must split it on the sequence axis.
    "video_keyframes_mask": torch.zeros(1, _VIDEO_TOKENS, 1),
  }
  inputs["video_keyframes_mask"][:, 0, :] = 1.0
  coords = torch.zeros(1, 3, _VIDEO_TOKENS, 2)
  for t in range(2):
    for h in range(2):
      for w in range(4):
        idx = t * 8 + h * 4 + w
        coords[:, 0, idx, :] = torch.tensor([float(t), float(t) + 1.0])
        coords[:, 1, idx, :] = torch.tensor([float(h), float(h) + 1.0])
        coords[:, 2, idx, :] = torch.tensor([float(w), float(w) + 1.0])
  inputs["video_coords"] = coords
  audio_coords = torch.zeros(1, 1, _AUDIO_TOKENS, 2)
  for i in range(_AUDIO_TOKENS):
    audio_coords[:, 0, i, :] = torch.tensor([float(i), float(i) + 0.5])
  inputs["audio_coords"] = audio_coords
  for key in ("hidden_states", "audio_hidden_states", "encoder_hidden_states",
              "audio_encoder_hidden_states"):
    inputs[key] = inputs[key].to(torch.bfloat16)
  return inputs


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _run_rank(rank: int, world: int, mode: str) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    import cache_dit
    from cache_dit import DBCacheConfig
    from cache_dit.caching.block_adapters import BlockAdapterRegister
    from cache_dit.distributed.config import ParallelismConfig

    inputs = _inputs()
    transformer = _build(device)

    reference = None
    if rank == 0:
      with torch.no_grad():
        reference = [
          t.float().cpu() for t in transformer(**_to_device(inputs, device), return_dict=False)
        ]
    torch.distributed.barrier()

    if mode == "cp":
      parallelism_config = ParallelismConfig(ulysses_size=world)
    else:
      parallelism_config = ParallelismConfig(tp_size=world)

    # Cache is disabled (warmup covers the whole run): this test isolates the parallel paths.
    cache_dit.enable_cache(
      BlockAdapterRegister.get_adapter(transformer),
      cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=100,
        max_cached_steps=0,
        num_inference_steps=8,
        enable_separate_cfg=False,
        residual_diff_threshold=0.0,
      ),
      parallelism_config=parallelism_config,
    )

    with torch.no_grad():
      out = [t.float().cpu() for t in transformer(**_to_device(inputs, device), return_dict=False)]

    if rank == 0:
      for name, got, ref in zip(("video", "audio"), out, reference):
        relative = (got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
        print(f"[ltx25-{mode}] {name} rel={relative.item():.6f} shape={tuple(got.shape)}")
        assert relative.item() < _REL_TOL, f"{mode} {name} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_ltx25_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode), nprocs=2, join=True)
