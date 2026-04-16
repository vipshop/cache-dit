import functools
from typing import Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

try:
  from nunchaku.models.transformers.transformer_flux_v2 import (
    NunchakuFluxAttention,
    NunchakuFluxFA2Processor,
  )
  from nunchaku.ops.fused import fused_qkv_norm_rottary
  from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenAttention,
    NunchakuQwenImageNaiveFA2Processor,
  )
  from nunchaku.models.transformers.transformer_zimage import (
    NunchakuZSingleStreamAttnProcessor,
    NunchakuZImageAttention,
  )
except ImportError:
  raise ImportError("NunchakuZImageTransformer2DModel, NunchakuFluxTransformer2DModelV2 and "
                    "NunchakuQwenImageTransformer2DModel requires the 'nunchaku' package. "
                    "Please install nunchaku>=1.10 before using the context parallelism for "
                    "nunchaku 4-bits models.")

from ...attention import _dispatch_attention_fn
from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("NunchakuFlux")
class NunchakuFluxContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    NunchakuFluxFA2Processor.__call__ = __patch_NunchakuFluxFA2Processor__call__
    # Also need to patch the parallel config and attention backend
    if not hasattr(NunchakuFluxFA2Processor, "_cp_config"):
      NunchakuFluxFA2Processor._cp_config = None
    if not hasattr(NunchakuFluxFA2Processor, "_attention_backend"):
      NunchakuFluxFA2Processor._attention_backend = None
    if not hasattr(NunchakuFluxAttention, "_cp_config"):
      NunchakuFluxAttention._cp_config = None
    if not hasattr(NunchakuFluxAttention, "_attention_backend"):
      NunchakuFluxAttention._attention_backend = None

    _cp_plan = {
      "": {
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "img_ids":
        _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        "txt_ids":
        _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@functools.wraps(NunchakuFluxFA2Processor.__call__)
def __patch_NunchakuFluxFA2Processor__call__(
  self: NunchakuFluxFA2Processor,
  attn: NunchakuFluxAttention,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor = None,
  **kwargs,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
  # The original implementation of NunchakuFluxFA2Processor.__call__
  # is not changed here for brevity. In actual implementation, we need to
  # modify the attention computation to support context parallelism.
  if attention_mask is not None:
    raise NotImplementedError("attention_mask is not supported")

  batch_size, _, channels = hidden_states.shape
  assert channels == attn.heads * attn.head_dim
  qkv = fused_qkv_norm_rottary(
    hidden_states,
    attn.to_qkv,
    attn.norm_q,
    attn.norm_k,
    (image_rotary_emb[0] if isinstance(image_rotary_emb, tuple) else image_rotary_emb),
  )

  if attn.added_kv_proj_dim is not None:
    assert encoder_hidden_states is not None
    assert isinstance(image_rotary_emb, tuple)
    qkv_context = fused_qkv_norm_rottary(
      encoder_hidden_states,
      attn.add_qkv_proj,
      attn.norm_added_q,
      attn.norm_added_k,
      image_rotary_emb[1],
    )
    qkv = torch.cat([qkv_context, qkv], dim=1)

  query, key, value = qkv.chunk(3, dim=-1)
  # Original implementation:
  # query = query.view(batch_size, -1, attn.heads, attn.head_dim).transpose(
  #     1, 2
  # )
  # key = key.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
  # value = value.view(batch_size, -1, attn.heads, attn.head_dim).transpose(
  #     1, 2
  # )
  # hidden_states = F.scaled_dot_product_attention(
  #     query,
  #     key,
  #     value,
  #     attn_mask=attention_mask,
  #     dropout_p=0.0,
  #     is_causal=False,
  # )
  # hidden_states = hidden_states.transpose(1, 2).reshape(
  #     batch_size, -1, attn.heads * attn.head_dim
  # )
  # hidden_states = hidden_states.to(query.dtype)

  # NOTE(DefTruth): Monkey patch to support context parallelism
  query = query.view(batch_size, -1, attn.heads, attn.head_dim)
  key = key.view(batch_size, -1, attn.heads, attn.head_dim)
  value = value.view(batch_size, -1, attn.heads, attn.head_dim)

  hidden_states = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    backend=getattr(self, "_attention_backend", None),
    cp_config=getattr(self, "_cp_config", None),
  )
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.to(query.dtype)

  if encoder_hidden_states is not None:
    encoder_hidden_states, hidden_states = (
      hidden_states[:, :encoder_hidden_states.shape[1]],
      hidden_states[:, encoder_hidden_states.shape[1]:],
    )
    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
    return hidden_states, encoder_hidden_states
  else:
    # for single transformer block, we split the proj_out into two linear layers
    hidden_states = attn.to_out(hidden_states)
    return hidden_states


@ContextParallelismPlannerRegister.register("NunchakuQwenImage")
class NunchakuQwenImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    NunchakuQwenImageNaiveFA2Processor.__call__ = (
      __patch_NunchakuQwenImageNaiveFA2Processor__call__)
    # Also need to patch the parallel config and attention backend
    if not hasattr(NunchakuQwenImageNaiveFA2Processor, "_cp_config"):
      NunchakuQwenImageNaiveFA2Processor._cp_config = None
    if not hasattr(NunchakuQwenImageNaiveFA2Processor, "_attention_backend"):
      NunchakuQwenImageNaiveFA2Processor._attention_backend = None
    if not hasattr(NunchakuQwenAttention, "_cp_config"):
      NunchakuQwenAttention._cp_config = None
    if not hasattr(NunchakuQwenAttention, "_attention_backend"):
      NunchakuQwenAttention._attention_backend = None

    _cp_plan = {
      "": {
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "pos_embed": {
        0: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        1: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@functools.wraps(NunchakuQwenImageNaiveFA2Processor.__call__)
def __patch_NunchakuQwenImageNaiveFA2Processor__call__(
  self: NunchakuQwenImageNaiveFA2Processor,
  attn: NunchakuQwenAttention,
  hidden_states: torch.FloatTensor,
  encoder_hidden_states: torch.FloatTensor = None,
  encoder_hidden_states_mask: torch.FloatTensor = None,
  attention_mask: Optional[torch.FloatTensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
  if encoder_hidden_states is None:
    raise ValueError("NunchakuQwenImageFA2Processor requires encoder_hidden_states (text stream)")

  seq_txt = encoder_hidden_states.shape[1]

  # Compute QKV for image stream (sample projections)
  img_qkv = attn.to_qkv(hidden_states)
  img_query, img_key, img_value = img_qkv.chunk(3, dim=-1)

  # Compute QKV for text stream (context projections)
  txt_qkv = attn.add_qkv_proj(encoder_hidden_states)
  txt_query, txt_key, txt_value = txt_qkv.chunk(3, dim=-1)

  # Reshape for multi-head attention
  img_query = img_query.unflatten(-1, (attn.heads, -1))  # [B, L, H, D]
  img_key = img_key.unflatten(-1, (attn.heads, -1))
  img_value = img_value.unflatten(-1, (attn.heads, -1))

  txt_query = txt_query.unflatten(-1, (attn.heads, -1))
  txt_key = txt_key.unflatten(-1, (attn.heads, -1))
  txt_value = txt_value.unflatten(-1, (attn.heads, -1))

  # Apply QK normalization
  assert attn.norm_q is not None
  img_query = attn.norm_q(img_query)
  assert attn.norm_k is not None
  img_key = attn.norm_k(img_key)
  assert attn.norm_added_q is not None
  txt_query = attn.norm_added_q(txt_query)
  assert attn.norm_added_k is not None
  txt_key = attn.norm_added_k(txt_key)

  # Apply rotary embeddings
  if image_rotary_emb is not None:
    img_freqs, txt_freqs = image_rotary_emb
    img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
    img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
    txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
    txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

  # Concatenate for joint attention: [text, image]
  joint_query = torch.cat([txt_query, img_query], dim=1)
  joint_key = torch.cat([txt_key, img_key], dim=1)
  joint_value = torch.cat([txt_value, img_value], dim=1)

  # Compute joint attention
  joint_hidden_states = _dispatch_attention_fn(
    joint_query,
    joint_key,
    joint_value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    # NOTE(DefTruth): Use the patched attention backend and
    # parallel config to make context parallelism work here.
    backend=getattr(self, "_attention_backend", None),
    cp_config=getattr(self, "_cp_config", None),
  )

  # Reshape back
  joint_hidden_states = joint_hidden_states.flatten(2, 3)
  joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

  # Split attention outputs back
  txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
  img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

  # Apply output projections
  img_attn_output = attn.to_out[0](img_attn_output)
  if len(attn.to_out) > 1:
    img_attn_output = attn.to_out[1](img_attn_output)  # dropout

  txt_attn_output = attn.to_add_out(txt_attn_output)

  return img_attn_output, txt_attn_output


@ContextParallelismPlannerRegister.register("NunchakuZImageTransformer2DModel")
class NunchakuZImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    if not hasattr(NunchakuZSingleStreamAttnProcessor, "_cp_config"):
      NunchakuZSingleStreamAttnProcessor._cp_config = None
    if not hasattr(NunchakuZSingleStreamAttnProcessor, "_attention_backend"):
      NunchakuZSingleStreamAttnProcessor._attention_backend = None
    if not hasattr(NunchakuZImageAttention, "_cp_config"):
      NunchakuZImageAttention._cp_config = None
    if not hasattr(NunchakuZImageAttention, "_attention_backend"):
      NunchakuZImageAttention._attention_backend = None

    n_noise_refiner_layers = len(transformer.noise_refiner)  # 2
    n_context_refiner_layers = len(transformer.context_refiner)  # 2
    n_layers = len(transformer.layers)  # 30
    # controlnet layer idx: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    # num_controlnet_samples = len(transformer.layers) // 2  # 15
    has_controlnet = kwargs.get("has_controlnet", None)
    if not has_controlnet:
      # cp plan for ZImageTransformer2DModel if no controlnet
      _cp_plan = {
        # 0. Hooks for noise_refiner layers, 2
        "noise_refiner.0": {
          "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "noise_refiner.*": {
          "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        f"noise_refiner.{n_noise_refiner_layers - 1}":
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
        # 1. Hooks for context_refiner layers, 2
        "context_refiner.0": {
          "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "context_refiner.*": {
          "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        f"context_refiner.{n_context_refiner_layers - 1}":
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
        # 2. Hooks for main transformer layers, num_layers=30
        "layers.0": {
          "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "layers.*": {
          "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        # `all_final_layer` is a ModuleDict and is supported by cache-dit's local CP runtime.
        "all_final_layer":
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
        # NOTE: The 'all_final_layer' is a ModuleDict of several final layers,
        # each for a specific patch size combination, so we do not add hooks for it here.
        # So, we have to gather the output of the last transformer layer.
        # f"layers.{num_layers - 1}": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      }
    else:
      # Special cp plan for NunchakuZImageTransformer2DModel with ZImageControlNetModel
      logger.warning("Using special context parallelism plan for NunchakuZImageTransformer2DModel "
                     "due to the 'has_controlnet' flag is set to True.")
      _cp_plan = {
        # zimage controlnet shared the same refiner as zimage, so, we need to
        # add gather hooks for all layers in noise_refiner and context_refiner.
        # 0. Hooks for noise_refiner layers, 2
        # Insert gather hook after each layers due to the ops: (controlnet)
        # - x = x + noise_refiner_block_samples[layer_idx]
        "noise_refiner.*": {
          "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        **{
          f"noise_refiner.{i}": _ContextParallelOutput(gather_dim=1, expected_dims=3)
          for i in range(n_noise_refiner_layers)
        },
        # 1. Hooks for context_refiner layers, 2
        "context_refiner.0": {
          "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "context_refiner.*": {
          "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        f"context_refiner.{n_context_refiner_layers - 1}":
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
        # 2. Hooks for main transformer layers, num_layers=30
        # Insert gather hook after each layers due to the ops: (main transformer)
        # - unified + controlnet_block_samples[layer_idx]
        "layers.*": {
          "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        **{
          f"layers.{i}": _ContextParallelOutput(gather_dim=1, expected_dims=3)
          for i in range(n_layers)
        },
        # `all_final_layer` is a ModuleDict and is supported by cache-dit's local CP runtime.
        "all_final_layer":
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
      }
    return _cp_plan
