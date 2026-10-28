"""Two-rank parity for the Ming-Image Bailing-MoE text-encoder TP plan."""
import os

import pytest
import torch
import torch.multiprocessing as mp

from cache_dit.models.pipelines.ming_image.mllm import MingImageMLLM

_HIDDEN = 32
_HEAD_DIM = 16
_NUM_HEADS = 4
_NUM_KV_HEADS = 2
_LAYERS = 4
_PORT = 29677


def _config():
  llm = {
    "hidden_size": _HIDDEN,
    "intermediate_size": 40,
    "moe_intermediate_size": 8,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
    "n_group": 2,
    "topk_group": 1,
    "routed_scaling_factor": 2.5,
    "num_hidden_layers": _LAYERS,
    "first_k_dense_replace": 1,
    "num_attention_heads": _NUM_HEADS,
    "num_key_value_heads": _NUM_KV_HEADS,
    "head_dim": _HEAD_DIM,
    "partial_rotary_factor": 0.5,
    "rope_theta": 600000.0,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "use_qkv_bias": False,
    "use_bias": False,
    "vocab_size": 64,
    "image_patch_token": 60,
    "image_start_token": 61,
    "image_end_token": 62,
    "video_patch_token": 63,
    "rope_scaling": {
      "type": "video_rope",
      "mrope_section": [1, 2, 1]
    },
  }
  vision = {
    "hidden_size": 32,
    "num_heads": 4,
    "intermediate_size": 64,
    "depth": 2,
    "in_chans": 3,
    "out_hidden_size": 16,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "hidden_act": "silu",
    "fullatt_block_indexes": [1],
    "window_size": 112,
  }
  return {"llm_config": llm, "vision_config": vision, "mlp_depth": 2}


def _run_rank(rank: int, world: int) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    from cache_dit.distributed.config import ParallelismConfig
    from cache_dit.distributed.text_encoders import parallelize_text_encoder

    torch.manual_seed(0)
    model = MingImageMLLM(_config(), img_gen_scales=(4, ), capture_layers=(1, 2, _LAYERS))
    model.to(device=device, dtype=torch.bfloat16).eval()
    generator = torch.Generator().manual_seed(3)
    embeds = torch.randn(1, 12, _HIDDEN, generator=generator).to(device=device,
                                                                 dtype=torch.bfloat16)
    position_ids = torch.zeros(3, 1, 12, dtype=torch.long, device=device)

    with torch.no_grad():
      reference = model.model.model(embeds, position_ids)[0].float().cpu()

    model = parallelize_text_encoder(model, ParallelismConfig(tp_size=world))
    with torch.no_grad():
      result = model.model.model(embeds, position_ids)[0].float().cpu()

    if rank == 0:
      cos = torch.nn.functional.cosine_similarity(reference.flatten(), result.flatten(),
                                                  dim=0).item()
      relative = (result - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)
      print(f"[ming-tep] cos={cos:.7f} rel={relative.item():.7f}")
      assert result.shape == reference.shape
      assert cos > 0.999, f"TE-P diverged (cos={cos:.7f})"
      assert relative.item() < 5e-3, f"TE-P diverged (rel={relative.item():.7f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_ming_image_text_encoder_tp_matches_single_rank():
  mp.spawn(_run_rank, args=(2, ), nprocs=2, join=True)
