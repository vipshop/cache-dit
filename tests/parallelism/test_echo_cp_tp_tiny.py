"""Multi-rank parity test for the Echo (JoyAI-Echo) CP and TP plumbing (tiny model, no weights).

Echo drives the LTX-2 backbone with a per-token *audio* timestep (`(batch, audio_tokens)`) and with
RoPE coordinates computed by the pipeline instead of inside the transformer. Both differ from the
regular LTX-2 pipelines the shared CP plan was written for:

- `audio_timestep` has to follow the sharded `audio_hidden_states`, which is the entry the LTX2 plan
  gained for Echo.
- `video_coords` / `audio_coords` stay full length: `rope` / `audio_rope` output-split them, so
  splitting the coordinates as well would split twice.

TP is shared with LTX-2 (per-head gated attention included).

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_echo_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

try:
  from diffusers.models.transformers.transformer_ltx2 import (  # noqa: F401
    LTX2PerturbedAttnProcessor, )
except ImportError:
  pytest.skip("diffusers without LTX-2 support", allow_module_level=True)

_VIDEO_TOKENS = 8
_AUDIO_TOKENS = 8
_TEXT_TOKENS = 4
_INNER_DIM = 32
_REL_TOL = 2e-2  # bf16 all-reduce order noise on a 2-layer random model
_PORT = 29572


def _build(device):
  from diffusers import LTX2VideoTransformer3DModel

  torch.manual_seed(0)
  transformer = LTX2VideoTransformer3DModel(
    in_channels=8,
    out_channels=8,
    patch_size=1,
    patch_size_t=1,
    num_attention_heads=4,
    attention_head_dim=8,
    cross_attention_dim=_INNER_DIM,
    vae_scale_factors=(1, 1, 1),
    pos_embed_max_pos=8,
    base_height=8,
    base_width=8,
    gated_attn=True,
    cross_attn_mod=True,
    audio_in_channels=8,
    audio_out_channels=8,
    audio_patch_size=1,
    audio_patch_size_t=1,
    audio_num_attention_heads=4,
    audio_attention_head_dim=8,
    audio_cross_attention_dim=_INNER_DIM,
    audio_scale_factor=1,
    audio_pos_embed_max_pos=8,
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
    use_keyframes_abs_pos_embedding=False,
  )
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs(transformer, device):
  generator = torch.Generator().manual_seed(3)
  coords = transformer.rope.prepare_video_coords(1, 2, 2, 2, device)
  audio_coords = transformer.audio_rope.prepare_audio_coords(1, _AUDIO_TOKENS, device)
  return {
    "hidden_states":
    torch.randn(1, _VIDEO_TOKENS, 8, generator=generator).to(torch.bfloat16),
    "audio_hidden_states":
    torch.randn(1, _AUDIO_TOKENS, 8, generator=generator).to(torch.bfloat16),
    "encoder_hidden_states":
    torch.randn(1, _TEXT_TOKENS, _INNER_DIM, generator=generator).to(torch.bfloat16),
    "audio_encoder_hidden_states":
    torch.randn(1, _TEXT_TOKENS, _INNER_DIM, generator=generator).to(torch.bfloat16),
    "timestep":
    torch.full((1, _VIDEO_TOKENS), 750.0),
    # Echo ships a per-token audio timestep instead of the scalar LTX-2 pipelines use.
    "audio_timestep":
    torch.full((1, _AUDIO_TOKENS), 500.0),
    "sigma":
    torch.full((1, ), 0.75),
    "audio_sigma":
    torch.full((1, ), 0.25),
    "video_coords":
    coords,
    "audio_coords":
    audio_coords,
    "fps":
    24.0,
    "use_cross_timestep":
    True,
  }


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

    transformer = _build(device)
    inputs = _inputs(transformer, device)

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
      for name, got, ref in (("video", out[0], reference[0]), ("audio", out[1], reference[1])):
        relative = (got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
        print(f"[echo-{mode}] {name} rel={relative.item():.6f} shape={tuple(got.shape)}")
        assert got.shape == ref.shape, f"{mode} {name} shape changed: {got.shape} vs {ref.shape}"
        assert relative.item() < _REL_TOL, f"{mode} {name} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_echo_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode), nprocs=2, join=True)
