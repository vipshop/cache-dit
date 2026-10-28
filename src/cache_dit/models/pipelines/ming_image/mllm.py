"""Minimal Ming-Image MLLM condition extractor.

Ming-Image conditions its diffusion transformer on two tensors produced by the Bailing-MoE MLLM
that ships in the ``mllm/`` component of the checkpoint:

* ``query_hidden_states`` — the learnable query block appended to the prompt tail, read out after
  the final norm: ``[B, num_query_tokens, hidden_size]``, i.e. 256 x 2048 for the release.
* ``direct_hidden_states`` — the prompt residual stream captured at layers
  ``selected_hidden_states_layers`` (``[5, 12, 20]``) and concatenated on the feature axis:
  ``[B, prompt_len, len(capture_layers) * hidden_size]``, i.e. ``[B, prompt_len, 6144]``. The query
  block is not part of it: the vendor code keeps only the original prompt prefix
  (``get_condition_embeds_for_image_gen`` slices ``hidden[:, :directvlm_max_valid_ind]``).

Only the inference closure is re-implemented here: no serving runtime, no fused MoE kernel, no
quantization, no KV cache, no logits. The forward path follows the vendor implementation
(``inclusionAI/Ming-Image``), which vLLM-Omni and SGLang ported as well, so the numerics stay
comparable with the served pipeline:

* the language stack is 20 x BailingMoeV2: layer 0 dense, layers 1..19 sparse with 256 experts and
  8 per token, dual router (``gate`` / ``image_gate``), group-limited top-k over 8 groups keeping 4,
  ``routed_scaling_factor = 2.5``, one shared expert;
* the router projects the fp32 post-attention norm, matching ``BailingMoeV2Gate``;
* patch tokens (reference image *and* the query block) are routed by ``image_gate``, the rest by
  ``gate``;
* positions are the 3-axis ``video_rope`` ids of ``get_t_scale_rope_index`` with
  ``scale_factor = 2.0``. The query block is treated as a ``1 x (2 * scale * scale)`` pseudo image,
  exactly like the vendor code that fuses the query tokens into ``image_grid_thw``;
* the vision tower is transformers' own ``Qwen2_5_VisionTransformerPretrainedModel`` (the
  checkpoint ships the Qwen2.5-VL keys verbatim), followed by ``linear_proj``
  (8192 -> 2048 -> 2048 with GELU) and ``F.normalize``, then scattered into the patch positions in
  prompt order;
* capture index ``i`` is the residual stream *entering* layer ``i`` (HF ``hidden_states[i]``),
  and index ``num_hidden_layers`` is the final norm output (HF ``hidden_states[-1]``, which is also
  where the query block is read from).

Prompt layout, from ``BailingMM2Processor.apply_chat_template`` plus the query suffix that
``append_input_ids_with_multiscale_learnable_tokens`` inserts before padding::

    "<role>SYSTEM</role>你是一个友好的AI助手。\\n\\ndetailed thinking off" <|role_end|>
    "<role>HUMAN</role>" ["<image>" "<imagePatch>" * n "</image>" "\\n"] prompt <|role_end|>
    "<role>ASSISTANT</role>" "<image>" "<imagePatch>" * 256 "</image>"
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, List, Tuple

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, Qwen2VLImageProcessor
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
  Qwen2_5_VisionTransformerPretrainedModel, )

from ....logger import init_logger

logger = init_logger(__name__)

_SYSTEM_PROMPT = "<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off"
_USER_PREFIX = "<role>HUMAN</role>"
_ASSISTANT_PREFIX = "<role>ASSISTANT</role>"
_IMAGE_PLACEHOLDER = "<IMAGE>"
_IMAGE_START_TOKEN = "<image>"
_IMAGE_PATCH_TOKEN = "<imagePatch>"
_IMAGE_END_TOKEN = "</image>"

_DEFAULT_CAPTURE_LAYERS = (5, 12, 20)
_DEFAULT_MROPE_SECTION = (8, 12, 12)
_DEFAULT_ROPE_SCALE_FACTOR = 2.0

#: Checkpoint tensors that are intentionally not loaded, with the reason.
_SKIPPED_WEIGHTS = (
  ("model.lm_head.weight", "the extractor never computes logits"),
  ("audio_gate", "Ming-Image has no audio modality, so the audio router is never applied"),
)


@dataclass
class MingMllmConditions:
  """Raw conditions consumed by ``MingImageConditioning``.

  :param query_hidden_states: final-norm hidden states of the query block,
    shape ``[B, num_query_tokens, hidden_size]``.
  :param direct_hidden_states: capture-layer hidden states of the prompt prefix (query block
    excluded), concatenated on the feature axis, shape
    ``[B, prompt_len, len(capture_layers) * hidden_size]``.
  """

  query_hidden_states: torch.Tensor
  direct_hidden_states: torch.Tensor


@dataclass
class MingMllmInputs:
  """Tokenised MLLM inputs of one image-generation prompt.

  :param input_ids: prompt ids including the appended query block, shape ``[1, S]``.
  :param attention_mask: valid-token mask, shape ``[1, S]``.
  :param position_ids: 3-axis ``video_rope`` positions, shape ``[3, 1, S]``.
  :param image_mask: patch-token mask driving the MoE image router, shape ``[1, S]``.
  :param pixel_values: reference-image patches ``[N, C * T * P * P]``, or ``None``.
  :param image_grid_thw: reference-image ``(t, h, w)`` grids ``[M, 3]``, or ``None``.
  :param prompt_length: length of the prompt prefix, i.e. ``S`` minus the query suffix.
  :param query_start: index of the first query token, i.e. ``S - 1 - num_query_tokens``.
  """

  input_ids: torch.Tensor
  attention_mask: torch.Tensor
  position_ids: torch.Tensor
  image_mask: torch.Tensor
  pixel_values: torch.Tensor | None
  image_grid_thw: torch.Tensor | None
  prompt_length: int
  query_start: int


def _to_namespace(value):
  """Recursively turn config dicts into attribute namespaces.

  :param value: dict, or any leaf value.
  :returns: the namespace for dicts, the value otherwise.
  """
  if isinstance(value, dict):
    return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
  return value


def _rope_settings(config) -> Tuple[str, Tuple[int, int, int]]:
  """Read the rope type and mRoPE split from a language config namespace.

  :param config: language config namespace.
  :returns: rope type (``video_rope`` for this checkpoint) and the T/H/W frequency split.
  """
  scaling = getattr(config, "rope_scaling", None)
  rope_type = None
  section = None
  if isinstance(scaling, dict):
    rope_type = scaling.get("rope_type") or scaling.get("type")
    section = scaling.get("mrope_section")
  elif scaling is not None:
    rope_type = getattr(scaling, "rope_type", None) or getattr(scaling, "type", None)
    section = getattr(scaling, "mrope_section", None)
  return rope_type or "default", tuple(section or _DEFAULT_MROPE_SECTION)


def format_generation_prompt(
  prompt: str,
  eos_token: str,
  has_reference_image: bool = False,
  system_prompt: str | None = None,
) -> str:
  """Build the Ming-Image single-turn prompt, mirroring ``apply_chat_template``.

  :param prompt: raw user prompt.
  :param eos_token: end-of-turn token, ``<|role_end|>`` for this checkpoint.
  :param has_reference_image: whether an ``<IMAGE>`` placeholder must open the human turn.
  :param system_prompt: overrides the default system sentence.
  :returns: the templated prompt, without the appended query suffix.
  """
  system = _SYSTEM_PROMPT
  if system_prompt is not None:
    system = _SYSTEM_PROMPT.replace("你是一个友好的AI助手。", system_prompt, 1)
  image_prefix = _IMAGE_PLACEHOLDER if has_reference_image else ""
  return system + eos_token + _USER_PREFIX + image_prefix + prompt + eos_token + _ASSISTANT_PREFIX


def expand_image_placeholder(text: str, grid_thw: torch.Tensor, merge_size: int) -> str:
  """Replace ``<IMAGE>`` with the ``<image>``/patch/``</image>`` block of one reference image.

  :param text: templated prompt holding at most one ``<IMAGE>`` placeholder.
  :param grid_thw: ``(t, h, w)`` grid of the reference image.
  :param merge_size: vision spatial merge size, 2 for this checkpoint.
  :returns: prompt text with the placeholder expanded, including the vendor's trailing newline.
  """
  num_patches = int(grid_thw.prod().item()) // (merge_size ** 2)
  replacement = _IMAGE_START_TOKEN + _IMAGE_PATCH_TOKEN * num_patches + _IMAGE_END_TOKEN + "\n"
  return text.replace(_IMAGE_PLACEHOLDER, replacement, 1)


def ming_rope_position_ids(
  input_ids: torch.Tensor,
  image_grid_thw: torch.Tensor | None,
  *,
  image_token_id: int,
  image_start_token_id: int,
  spatial_merge_size: int,
  scale_factor: float = _DEFAULT_ROPE_SCALE_FACTOR,
) -> torch.Tensor:
  """Port of ``get_t_scale_rope_index`` for image-only prompts.

  Text tokens share one position across the three axes; each image block gets
  ``t = t_index * scale_factor + text_len``, ``h = row - (h - 1) // 2 + t`` and
  ``w = col - (w - 1) // 2 + t``, and the following text resumes at ``max(t) + 1``.

  :param input_ids: token ids, shape ``[B, S]``.
  :param image_grid_thw: grid per image block in prompt order, ``[M, 3]``; ``None`` for text-only.
  :param image_token_id: ``<imagePatch>`` id, the first token of every image block.
  :param image_start_token_id: ``<image>`` id, used to count image blocks.
  :param spatial_merge_size: vision spatial merge size.
  :param scale_factor: temporal stride multiplier, 2.0 for Ming.
  :returns: position ids, shape ``[3, B, S]``.
  """
  batch, seq_len = input_ids.shape
  position_ids = torch.ones(3, batch, seq_len, dtype=torch.long, device=input_ids.device)
  image_index = 0
  for sample in range(batch):
    tokens = input_ids[sample].tolist()
    image_nums = sum(1 for index, token in enumerate(tokens)
                     if token == image_start_token_id and index +
                     1 < len(tokens) and tokens[index + 1] == image_token_id)
    blocks: List[torch.Tensor] = []
    st = 0
    for _ in range(image_nums):
      if image_grid_thw is None:
        raise ValueError("Ming-Image prompt contains image blocks but no image_grid_thw")
      ed = tokens.index(image_token_id, st)
      grid_t, grid_h, grid_w = (int(value) for value in image_grid_thw[image_index])
      image_index += 1
      llm_grid_h = grid_h // spatial_merge_size
      llm_grid_w = grid_w // spatial_merge_size
      text_len = ed - st
      st_idx = float(blocks[-1][0].max().item()) + 1 if blocks else 0.0
      blocks.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)
      t_index = torch.arange(grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
      h_index = (torch.arange(llm_grid_h).view(1, -1, 1).expand(grid_t, -1, llm_grid_w).flatten() -
                 (llm_grid_h - 1) // 2)
      w_index = (torch.arange(llm_grid_w).view(1, 1, -1).expand(grid_t, llm_grid_h, -1).flatten() -
                 (llm_grid_w - 1) // 2)
      t_index = t_index * scale_factor + text_len + st_idx
      blocks.append(torch.stack([t_index, h_index + t_index, w_index + t_index]))
      st = ed + grid_t * llm_grid_h * llm_grid_w
    if st < seq_len:
      st_idx = float(blocks[-1][0].max().item()) + 1 if blocks else 0.0
      blocks.append(torch.arange(seq_len - st).view(1, -1).expand(3, -1) + st_idx)
    position_ids[:, sample, :] = torch.cat(blocks, dim=1).long()
  return position_ids


def apply_3d_rotary_pos_emb(
  query: torch.Tensor,
  key: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  mrope_section: Tuple[int, int, int] = _DEFAULT_MROPE_SECTION,
  rope_type: str = "video_rope",
) -> Tuple[torch.Tensor, torch.Tensor]:
  """Port of Ming's ``apply_3d_rotary_pos_emb`` with partial rotary applied afterwards.

  ``video_rope`` interleaves the H/W sections element-wise and moves the temporal section to the
  tail, unlike the standard mRoPE layout that keeps T/H/W as contiguous chunks.

  :param query: query states ``[B, H, S, D]``.
  :param key: key states ``[B, H_kv, S, D]``.
  :param cos: cosine table ``[3, B, S, D_rope]``.
  :param sin: sine table ``[3, B, S, D_rope]``.
  :param mrope_section: frequency split for the temporal/height/width axes.
  :param rope_type: ``video_rope`` (Ming) or ``m_rope``/``3D`` (Qwen2.5-VL layout).
  :returns: rotated query and key.
  """
  if rope_type == "video_rope":
    sections = list(mrope_section)
    sections = [sections[0], sections[1] + sections[2]]
    sections = (sections * 2)[::-1]
    index = 0
    cos_parts: List[torch.Tensor] = []
    sin_parts: List[torch.Tensor] = []
    for position, section in enumerate(sections):
      if position % 2 == 0:
        for step in range(section):
          axis = 1 if step % 2 == 0 else 2
          cos_parts.append(cos[axis, ..., index:index + 1])
          sin_parts.append(sin[axis, ..., index:index + 1])
          index += 1
      else:
        cos_parts.append(cos[0, ..., index:index + section])
        sin_parts.append(sin[0, ..., index:index + section])
        index += section
    cos = torch.cat(cos_parts, dim=-1).unsqueeze(1)
    sin = torch.cat(sin_parts, dim=-1).unsqueeze(1)
  elif rope_type in ("m_rope", "3D"):
    sections = list(mrope_section) * 2
    cos = torch.cat([part[axis % 3] for axis, part in enumerate(cos.split(sections, dim=-1))],
                    dim=-1).unsqueeze(1)
    sin = torch.cat([part[axis % 3] for axis, part in enumerate(sin.split(sections, dim=-1))],
                    dim=-1).unsqueeze(1)
  else:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

  rotary_dim = cos.shape[-1]

  def rotate(states: torch.Tensor) -> torch.Tensor:
    rotated_part, tail = states[..., :rotary_dim], states[..., rotary_dim:]
    half = rotary_dim // 2
    half_rotated = torch.cat((-rotated_part[..., half:], rotated_part[..., :half]), dim=-1)
    return torch.cat((rotated_part * cos + half_rotated * sin, tail), dim=-1)

  return rotate(query), rotate(key)


class MingRMSNorm(nn.Module):
  """BailingMoeV2 RMSNorm: fp32 statistics, dtype-preserving weight multiply."""

  def __init__(self, hidden_size: int, eps: float = 1e-6):
    """
    :param hidden_size: normalized dimension.
    :param eps: variance epsilon.
    """
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    :param hidden_states: input states.
    :returns: normalized states in the input dtype.
    """
    input_dtype = hidden_states.dtype
    states = hidden_states.to(torch.float32)
    variance = states.pow(2).mean(-1, keepdim=True)
    states = states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * states.to(input_dtype)


class MingRotaryEmbedding(nn.Module):
  """3-axis RoPE tables for the ``video_rope`` layout."""

  def __init__(self, head_dim: int, rope_theta: float):
    """
    :param head_dim: attention head dimension; the table covers the full head width and the
      partial-rotary truncation happens in :func:`apply_3d_rotary_pos_emb`.
    :param rope_theta: RoPE base.
    """
    super().__init__()
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)

  def forward(self, hidden_states: torch.Tensor,
              position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param hidden_states: states used only for device and dtype.
    :param position_ids: 3-axis positions, shape ``[3, B, S]``.
    :returns: cosine and sine tables, shape ``[3, B, S, head_dim]`` in the input dtype.
    """
    inv_freq = self.inv_freq[None, None, :, None].float().to(hidden_states.device)
    positions = position_ids[:, :, None, :].float()
    with torch.autocast(hidden_states.device.type, enabled=False):
      freqs = (inv_freq @ positions).transpose(2, 3)
      emb = torch.cat((freqs, freqs), dim=-1)
      cos = emb.cos()
      sin = emb.sin()
    return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)


class MingMLP(nn.Module):
  """BailingMoeV2 gated MLP, used by the dense layer and by every expert."""

  def __init__(self, hidden_size: int, intermediate_size: int):
    """
    :param hidden_size: input and output dimension.
    :param intermediate_size: gated intermediate dimension.
    """
    super().__init__()
    self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    :param hidden_states: input states ``[..., hidden_size]``.
    :returns: MLP output, same shape.
    """
    return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class MingRouter(nn.Module):
  """BailingMoeV2 router: sigmoid scores, group-limited top-k, rescaled weights."""

  def __init__(
    self,
    hidden_size: int,
    num_experts: int,
    top_k: int,
    n_group: int,
    topk_group: int,
    routed_scaling_factor: float,
  ):
    """
    :param hidden_size: router input dimension.
    :param num_experts: total routed experts.
    :param top_k: experts kept per token.
    :param n_group: number of expert groups.
    :param topk_group: groups kept per token.
    :param routed_scaling_factor: applied after renormalising the kept scores.
    """
    super().__init__()
    self.num_experts = num_experts
    self.top_k = top_k
    self.n_group = n_group
    self.topk_group = topk_group
    self.routed_scaling_factor = routed_scaling_factor
    self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
    self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

  def group_limited_topk(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep the top-k experts of the best-scoring groups.

    :param scores: routing scores with the expert bias applied, ``[N, num_experts]``.
    :returns: kept scores and expert indices, both ``[N, top_k]``.
    """
    num_tokens = scores.shape[0]
    group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_index = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_index, 1)
    score_mask = (group_mask.unsqueeze(-1).expand(num_tokens, self.n_group,
                                                  self.num_experts // self.n_group).reshape(
                                                    num_tokens, -1))
    masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    return torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)

  def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    :param hidden_states: router input ``[..., hidden_size]``.
    :returns: expert indices ``[N, top_k]`` (int64) and weights ``[N, top_k]`` (fp32).
    """
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    scores = torch.sigmoid(F.linear(flat.float(), self.weight.float()))
    _, topk_index = self.group_limited_topk(scores + self.expert_bias.float())
    topk_score = torch.gather(scores, dim=1, index=topk_index)
    if self.top_k > 1:
      topk_score = topk_score / (topk_score.sum(dim=-1, keepdim=True) + 1e-20)
    return topk_index, topk_score * self.routed_scaling_factor


class MingAttention(nn.Module):
  """BailingMoeV2 attention: fused QKV, per-head QK norm, partial 3-axis RoPE."""

  def __init__(self, config: SimpleNamespace, mrope_section, rope_type: str):
    """
    :param config: language config namespace.
    :param mrope_section: RoPE frequency split for the T/H/W axes.
    :param rope_type: ``video_rope`` for this checkpoint.
    """
    super().__init__()
    self.num_heads = config.num_attention_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.head_dim = config.head_dim or config.hidden_size // self.num_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.mrope_section = tuple(mrope_section)
    self.rope_type = rope_type
    self.query_key_value = nn.Linear(
      config.hidden_size,
      (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
      bias=bool(getattr(config, "use_qkv_bias", False)),
    )
    self.q_norm = MingRMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.k_norm = MingRMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.dense = nn.Linear(
      self.num_heads * self.head_dim,
      config.hidden_size,
      bias=bool(getattr(config, "use_bias", False)),
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """
    :param hidden_states: input states ``[B, S, hidden_size]``.
    :param cos: RoPE cosine table ``[3, B, S, head_dim]``.
    :param sin: RoPE sine table ``[3, B, S, head_dim]``.
    :param attn_mask: boolean keep-mask ``[B, 1, S, S]`` when padding must be masked; ``None``
      selects plain causal attention.
    :returns: attention output ``[B, S, hidden_size]``.
    """
    batch, seq_len, _ = hidden_states.shape
    qkv = self.query_key_value(hidden_states).view(batch, seq_len,
                                                   self.num_heads + 2 * self.num_key_value_heads,
                                                   self.head_dim)
    query, key, value = qkv.split(
      [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2)
    query = self.q_norm(query.transpose(1, 2))
    key = self.k_norm(key.transpose(1, 2))
    value = value.transpose(1, 2)
    query, key = apply_3d_rotary_pos_emb(query, key, cos, sin, self.mrope_section, self.rope_type)
    if self.num_key_value_groups > 1:
      key = key.repeat_interleave(self.num_key_value_groups, dim=1)
      value = value.repeat_interleave(self.num_key_value_groups, dim=1)
    attn_output = F.scaled_dot_product_attention(
      query,
      key,
      value,
      attn_mask=attn_mask,
      dropout_p=0.0,
      is_causal=attn_mask is None,
    )
    attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
    return self.dense(attn_output)


class MingMoeBlock(nn.Module):
  """BailingMoeV2 sparse block with the MultiRouter (text + image) gate."""

  def __init__(self, config: SimpleNamespace):
    """
    :param config: language config namespace.
    """
    super().__init__()
    self.num_experts = config.num_experts
    self.top_k = config.num_experts_per_tok
    self.experts = nn.ModuleList(
      MingMLP(config.hidden_size, config.moe_intermediate_size) for _ in range(config.num_experts))
    num_shared_experts = getattr(config, "num_shared_experts", None) or 0
    self.shared_experts = (MingMLP(config.hidden_size, config.moe_intermediate_size *
                                   num_shared_experts) if num_shared_experts > 0 else None)
    router_kwargs = {
      "hidden_size": config.hidden_size,
      "num_experts": config.num_experts,
      "top_k": config.num_experts_per_tok,
      "n_group": config.n_group,
      "topk_group": config.topk_group,
      "routed_scaling_factor": config.routed_scaling_factor,
    }
    self.gate = MingRouter(**router_kwargs)
    self.image_gate = MingRouter(**router_kwargs)

  def _combine(self, flat: torch.Tensor, topk_index: torch.Tensor,
               topk_weight: torch.Tensor) -> torch.Tensor:
    """Run the kept experts and combine their outputs with the routing weights.

    :param flat: tokens ``[N, hidden_size]``.
    :param topk_index: expert indices ``[N, top_k]``.
    :param topk_weight: routed weights ``[N, top_k]`` in fp32.
    :returns: combined expert output ``[N, hidden_size]`` in the input dtype.
    """
    order = topk_index.reshape(-1).argsort(stable=True)
    token_of = order // self.top_k
    counts = torch.bincount(topk_index.reshape(-1), minlength=self.num_experts).tolist()
    outputs = []
    start = 0
    for expert_id, count in enumerate(counts):
      if count == 0:
        continue
      tokens = flat.index_select(0, token_of[start:start + count])
      outputs.append(self.experts[expert_id](tokens))
      start += count
    if not outputs:
      return torch.zeros_like(flat)
    stacked = torch.cat(outputs, dim=0)
    merged = flat.new_empty(stacked.shape)
    merged[order] = stacked
    combined = merged.view(flat.shape[0], self.top_k, -1).float() * topk_weight.unsqueeze(-1)
    return combined.sum(dim=1).to(flat.dtype)

  def forward(self, hidden_states: torch.Tensor, image_mask: torch.Tensor | None) -> torch.Tensor:
    """
    :param hidden_states: post-attention norm ``[B, S, hidden_size]``.
    :param image_mask: patch-token mask ``[B, S]`` selecting the image router, or ``None``.
    :returns: block output, same shape as the input.
    """
    batch, seq_len, hidden_size = hidden_states.shape
    flat = hidden_states.reshape(-1, hidden_size)
    topk_index, topk_weight = self.gate(flat)
    if image_mask is not None:
      image_index, image_weight = self.image_gate(flat)
      patch = image_mask.reshape(-1, 1)
      topk_index = torch.where(patch, image_index, topk_index)
      topk_weight = torch.where(patch, image_weight, topk_weight)
    output = self._combine(flat, topk_index, topk_weight).view(batch, seq_len, hidden_size)
    if self.shared_experts is not None:
      output = output + self.shared_experts(hidden_states)
    return output


class MingDecoderLayer(nn.Module):
  """BailingMoeV2 decoder layer; dense MLP below ``first_k_dense_replace``, MoE above."""

  def __init__(self, config: SimpleNamespace, layer_idx: int, mrope_section, rope_type: str):
    """
    :param config: language config namespace.
    :param layer_idx: layer index, which decides dense versus sparse MLP.
    :param mrope_section: RoPE frequency split.
    :param rope_type: ``video_rope`` for this checkpoint.
    """
    super().__init__()
    self.attention = MingAttention(config, mrope_section, rope_type)
    self.is_moe = config.num_experts is not None and layer_idx >= config.first_k_dense_replace
    self.mlp = (MingMoeBlock(config)
                if self.is_moe else MingMLP(config.hidden_size, config.intermediate_size))
    self.input_layernorm = MingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.post_attention_layernorm = MingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

  def forward(
    self,
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    attn_mask: torch.Tensor | None,
    image_mask: torch.Tensor | None,
  ) -> torch.Tensor:
    """
    :param hidden_states: layer input ``[B, S, hidden_size]``.
    :param cos: RoPE cosine table.
    :param sin: RoPE sine table.
    :param attn_mask: boolean keep-mask, or ``None`` for plain causal attention.
    :param image_mask: patch-token mask for the MoE image router.
    :returns: layer output ``[B, S, hidden_size]``.
    """
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.attention(hidden_states, cos, sin, attn_mask)
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = (self.mlp(hidden_states, image_mask)
                     if self.is_moe else self.mlp(hidden_states))
    return residual + hidden_states


class MingBailingModel(nn.Module):
  """BailingMoeV2 trunk: embeddings, decoder layers and the final norm."""

  def __init__(self, config: SimpleNamespace):
    """
    :param config: language config namespace.
    """
    super().__init__()
    self.rope_type, self.mrope_section = _rope_settings(config)
    head_dim = config.head_dim or config.hidden_size // config.num_attention_heads
    if self.rope_type in ("video_rope", "m_rope", "3D"):
      rotary_dim = 2 * sum(self.mrope_section)
      expected = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
      if rotary_dim != expected:
        raise ValueError(f"mrope_section {self.mrope_section} covers {rotary_dim} rotary dims, but "
                         f"head_dim={head_dim} with partial_rotary_factor="
                         f"{getattr(config, 'partial_rotary_factor', 1.0)} expects {expected}")
    self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
    self.layers = nn.ModuleList(
      MingDecoderLayer(config, layer_idx, self.mrope_section, self.rope_type)
      for layer_idx in range(config.num_hidden_layers))
    self.norm = MingRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.rotary_emb = MingRotaryEmbedding(head_dim, config.rope_theta)

  def forward(
      self,
      inputs_embeds: torch.Tensor,
      position_ids: torch.Tensor,
      image_mask: torch.Tensor | None = None,
      attn_mask: torch.Tensor | None = None,
      capture_layers: Tuple[int, ...] = (),
  ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Run the trunk and capture the residual stream at the requested depths.

    :param inputs_embeds: fused input embeddings ``[B, S, hidden_size]``.
    :param position_ids: 3-axis RoPE positions ``[3, B, S]``.
    :param image_mask: patch-token mask for the MoE image router.
    :param attn_mask: boolean keep-mask, or ``None`` for plain causal attention.
    :param capture_layers: layer indices captured on entry; ``num_hidden_layers`` captures the
      final norm output, matching HF ``hidden_states[-1]``.
    :returns: final normed states and the captured tensors keyed by layer index.
    """
    capture = set(capture_layers)
    captured: Dict[int, torch.Tensor] = {}
    cos, sin = self.rotary_emb(inputs_embeds, position_ids)
    hidden_states = inputs_embeds
    for layer_idx, layer in enumerate(self.layers):
      if layer_idx in capture:
        captured[layer_idx] = hidden_states
      hidden_states = layer(hidden_states, cos, sin, attn_mask, image_mask)
    hidden_states = self.norm(hidden_states)
    if len(self.layers) in capture:
      captured[len(self.layers)] = hidden_states
    return hidden_states, captured


class MingBailingForCausalLM(nn.Module):
  """Checkpoint-compatible container so the keys keep their ``model.model.*`` prefix."""

  def __init__(self, config: SimpleNamespace):
    """
    :param config: language config namespace.
    """
    super().__init__()
    self.model = MingBailingModel(config)


def _skip_reason(name: str) -> str | None:
  """Return why a checkpoint tensor is not needed, or ``None`` when it must be loaded.

  :param name: checkpoint tensor name.
  :returns: skip reason, or ``None``.
  """
  for pattern, reason in _SKIPPED_WEIGHTS:
    if pattern in name:
      return reason
  return None


def iter_checkpoint_tensors(directory: Path) -> Iterator[Tuple[str, torch.Tensor]]:
  """Yield ``(name, tensor)`` pairs of a safetensors checkpoint, one tensor at a time.

  :param directory: directory holding ``model.safetensors`` or its sharded index.
  :returns: iterator over the checkpoint tensors.
  """
  index_path = directory / "model.safetensors.index.json"
  if index_path.is_file():
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shards: Dict[str, List[str]] = {}
    for name, shard in weight_map.items():
      shards.setdefault(shard, []).append(name)
    for shard, names in sorted(shards.items()):
      with safe_open(str(directory / shard), framework="pt", device="cpu") as handle:
        for name in names:
          yield name, handle.get_tensor(name)
    return
  single = directory / "model.safetensors"
  if not single.is_file():
    raise FileNotFoundError(f"No safetensors checkpoint found under {directory}")
  with safe_open(str(single), framework="pt", device="cpu") as handle:
    for name in handle.keys():
      yield name, handle.get_tensor(name)


def load_ming_mllm_weights(model: MingImageMLLM, directory: Path) -> Dict[str, str]:
  """Load the MLLM shards with explicit key accounting.

  Every checkpoint tensor is either copied into a parameter of the same shape or recorded as an
  intentionally skipped key; unknown keys, shape mismatches and untouched parameters all raise.

  :param model: target module.
  :param directory: ``mllm/`` component directory.
  :returns: mapping of skipped checkpoint key to the reason it is not needed.
  """
  params = dict(model.named_parameters())
  loaded = set()
  skipped: Dict[str, str] = {}
  for name, tensor in iter_checkpoint_tensors(directory):
    reason = _skip_reason(name)
    if reason is not None:
      skipped[name] = reason
      continue
    param = params.get(name)
    if param is None:
      raise KeyError(f"Unexpected Ming-Image MLLM checkpoint tensor: {name}")
    if tuple(param.shape) != tuple(tensor.shape):
      raise ValueError(f"Ming-Image MLLM shape mismatch for {name}: "
                       f"checkpoint {tuple(tensor.shape)} vs model {tuple(param.shape)}")
    with torch.no_grad():
      param.copy_(tensor)
    loaded.add(name)
  missing = sorted(name for name in params
                   if name not in loaded and not name.startswith("query_tokens_dict."))
  if missing:
    raise ValueError(f"Ming-Image MLLM weights missing from {directory}: {missing[:8]}")
  if skipped:
    logger.info(
      "Ming-Image MLLM: loaded %d tensors, skipped %d (%s)",
      len(loaded),
      len(skipped),
      "; ".join(sorted(set(skipped.values()))),
    )
  return skipped


def load_query_tokens(model: MingImageMLLM, mlp_directory: Path) -> None:
  """Load the learnable image-generation query tokens from ``mlp/model.safetensors``.

  :param model: target module holding ``query_tokens_dict``.
  :param mlp_directory: ``mlp/`` component directory.
  """
  path = mlp_directory / "model.safetensors"
  if not path.is_file():
    raise FileNotFoundError(f"Missing Ming-Image query tokens: {path}")
  with safe_open(str(path), framework="pt", device="cpu") as handle:
    available = set(handle.keys())
    for scale_name, param in model.query_tokens_dict.items():
      key = f"query_tokens_dict.{scale_name}"
      if key not in available:
        raise ValueError(f"Missing Ming-Image query tokens {key!r} in {path}")
      tensor = handle.get_tensor(key)
      if tuple(tensor.shape) != tuple(param.shape):
        raise ValueError(f"Ming-Image query token shape mismatch for {key}: "
                         f"checkpoint {tuple(tensor.shape)} vs model {tuple(param.shape)}")
      with torch.no_grad():
        param.copy_(tensor)


class MingImageMLLM(nn.Module):
  """Ming-Image MLLM that turns a prompt (plus optional reference image) into DiT conditions.

  The module owns the vision tower, the linear projector, the query tokens and the BailingMoeV2
  trunk; the connector, the diffusion transformer and the VAE live elsewhere.
  """

  def __init__(
    self,
    config: dict,
    *,
    img_gen_scales: Tuple[int, ...] = (16, ),
    capture_layers: Tuple[int, ...] | None = None,
    max_num_images: int = 1,
  ):
    """
    :param config: parsed ``mllm/config.json`` (``llm_config``, ``vision_config``, ``mlp_depth``).
    :param img_gen_scales: per-scale query grids; ``(16,)`` yields the released 256 tokens.
    :param capture_layers: depths captured for the direct-VLM condition; defaults to
      ``(5, 12, num_hidden_layers)``.
    :param max_num_images: reference images allowed per prompt.
    """
    super().__init__()
    self.config = config
    llm_config = _to_namespace(config["llm_config"])
    vision_config = _to_namespace(config["vision_config"])
    self.llm_config = llm_config
    self.vision_config = vision_config
    self.mlp_depth = int(config.get("mlp_depth", 2))

    num_layers = llm_config.num_hidden_layers
    self.capture_layers = tuple(
      capture_layers if capture_layers is not None else _DEFAULT_CAPTURE_LAYERS)
    invalid = [layer for layer in self.capture_layers if not 0 <= layer <= num_layers]
    if invalid:
      raise ValueError(
        f"capture layers {invalid} are outside [0, {num_layers}] for a {num_layers}-layer trunk")

    self.image_patch_token = int(llm_config.image_patch_token)
    self.image_start_token = int(llm_config.image_start_token)
    self.image_end_token = int(llm_config.image_end_token)
    self.spatial_merge_size = int(vision_config.spatial_merge_size)
    self.max_num_images = int(max_num_images)

    self.model = MingBailingForCausalLM(llm_config)

    vision_kwargs = Qwen2_5_VLVisionConfig.from_dict(dict(config["vision_config"]))
    vision_kwargs._attn_implementation = "sdpa"
    self.vision = Qwen2_5_VisionTransformerPretrainedModel(vision_kwargs)

    projector = [nn.Linear(vision_config.out_hidden_size, llm_config.hidden_size)]
    for _ in range(1, self.mlp_depth):
      projector.append(nn.GELU())
      projector.append(nn.Linear(llm_config.hidden_size, llm_config.hidden_size))
    self.linear_proj = nn.Sequential(*projector)

    self.img_gen_scales = tuple(int(scale) for scale in img_gen_scales)
    if not self.img_gen_scales:
      raise ValueError("img_gen_scales must not be empty")
    self.query_tokens_dict = nn.ParameterDict({
      f"{scale}x{scale}":
      nn.Parameter(F.normalize(torch.randn(scale * scale, llm_config.hidden_size), dim=-1))
      for scale in self.img_gen_scales
    })

    self.tokenizer = None
    self.image_processor = None

  @property
  def dtype(self) -> torch.dtype:
    """:returns: parameter dtype of the module."""
    return self.model.model.word_embeddings.weight.dtype

  @property
  def device(self) -> torch.device:
    """:returns: device of the module parameters."""
    return self.model.model.word_embeddings.weight.device

  @property
  def direct_hidden_size(self) -> int:
    """:returns: direct-condition width, ``len(capture_layers) * hidden_size`` (6144 for the
      release, i.e. the channel count ``proj_directvlm`` expects)."""
    return len(self.capture_layers) * self.llm_config.hidden_size

  @property
  def num_query_tokens(self) -> int:
    """:returns: width of the query block consumed by the connector (256 for the release)."""
    scale = self.img_gen_scales[-1]
    return scale * scale

  @property
  def suffix_length(self) -> int:
    """:returns: tokens appended after the prompt: query blocks plus their delimiters."""
    return sum(scale * scale + 2 for scale in self.img_gen_scales)

  def query_block(self) -> torch.Tensor:
    """:returns: concatenated learnable query tokens ``[num_query_total, hidden_size]``."""
    return torch.cat([param for param in self.query_tokens_dict.values()], dim=0)

  def query_suffix_ids(self) -> List[int]:
    """:returns: token ids of the appended ``<image>``/patch/``</image>`` blocks, per scale."""
    suffix: List[int] = []
    for scale in self.img_gen_scales:
      suffix.append(self.image_start_token)
      suffix.extend([self.image_patch_token] * (scale * scale))
      suffix.append(self.image_end_token)
    return suffix

  @classmethod
  def from_pretrained(
    cls,
    pretrained_model_name_or_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    torch_dtype: torch.dtype | None = torch.bfloat16,
    dtype: torch.dtype | None = None,
    capture_layers: Tuple[int, ...] | None = None,
    max_num_images: int = 1,
    local_files_only: bool = True,
    load_tokenizer: bool = True,
  ) -> MingImageMLLM:
    """Load the MLLM component of a Ming-Image checkpoint.

    :param pretrained_model_name_or_path: checkpoint root (holding ``mllm/`` and ``mlp/``) or the
      ``mllm/`` component directory itself.
    :param device: target device.
    :param torch_dtype: parameter dtype; the released checkpoint is bf16.
    :param dtype: alias of ``torch_dtype``.
    :param capture_layers: overrides the default ``(5, 12, num_hidden_layers)`` depths.
    :param max_num_images: reference images allowed per prompt.
    :param local_files_only: never reach the network for the tokenizer and image processor.
    :param load_tokenizer: also build the tokenizer and image processor.
    :returns: the loaded module in eval mode.
    """
    path = Path(pretrained_model_name_or_path)
    if (path / "mllm" / "config.json").is_file():
      component_root = path
      mllm_directory = path / "mllm"
    else:
      component_root = path.parent
      mllm_directory = path
    config = json.loads((mllm_directory / "config.json").read_text())
    mlp_config_path = component_root / "mlp" / "config.json"
    if not mlp_config_path.is_file():
      raise FileNotFoundError(f"Missing Ming-Image mlp config: {mlp_config_path}")
    mlp_config = json.loads(mlp_config_path.read_text())
    img_gen_scales = tuple(mlp_config.get("img_gen_scales", (16, )))
    if capture_layers is None:
      selected = mlp_config.get("selected_hidden_states_layers")
      capture_layers = tuple(selected) if selected else None

    model_dtype = torch_dtype if torch_dtype is not None else dtype
    model = cls(
      config,
      img_gen_scales=img_gen_scales,
      capture_layers=capture_layers,
      max_num_images=max_num_images,
    )
    model.to(device=device, dtype=model_dtype or torch.bfloat16)
    load_ming_mllm_weights(model, mllm_directory)
    load_query_tokens(model, component_root / "mlp")
    model.requires_grad_(False)
    model.eval()
    if load_tokenizer:
      model.tokenizer = AutoTokenizer.from_pretrained(str(mllm_directory),
                                                      local_files_only=local_files_only)
      model.image_processor = Qwen2VLImageProcessor.from_pretrained(
        str(mllm_directory), local_files_only=local_files_only)
    return model

  def _image_features(self, pixel_values: torch.Tensor | None,
                      grid_thw: torch.Tensor | None) -> torch.Tensor | None:
    """Encode and normalise the reference-image patches.

    :param pixel_values: flattened patches ``[N, C * T * P * P]``, or ``None``.
    :param grid_thw: ``(t, h, w)`` grids ``[M, 3]``, or ``None``.
    :returns: L2-normalised features ``[N / merge^2, hidden_size]``, or ``None`` without an image.
    """
    if pixel_values is None or grid_thw is None:
      return None
    dtype = self.dtype
    outputs = self.vision(pixel_values.to(dtype), grid_thw=grid_thw.to(pixel_values.device))
    if isinstance(outputs, torch.Tensor):
      merged = outputs
    else:
      merged = getattr(outputs, "pooler_output", None)
      if merged is None:
        merged = outputs.last_hidden_state
    return F.normalize(self.linear_proj(merged.to(dtype)), dim=-1)

  def build_inputs_embeds(self, inputs: MingMllmInputs) -> torch.Tensor:
    """Fuse text embeddings, image features and query tokens into the trunk input.

    Patch positions are filled in prompt order: reference-image blocks first, then the learnable
    query block.

    :param inputs: prepared inputs.
    :returns: input embeddings ``[1, S, hidden_size]``.
    """
    device = self.model.model.word_embeddings.weight.device
    embeds = self.model.model.word_embeddings(inputs.input_ids.to(device))
    patch_mask = inputs.image_mask.to(device)
    num_patch = int(patch_mask.sum())
    if num_patch == 0:
      return embeds
    features = self._image_features(
      None if inputs.pixel_values is None else inputs.pixel_values.to(device),
      None if inputs.image_grid_thw is None else inputs.image_grid_thw.to(device),
    )
    query = self.query_block().to(device=device, dtype=embeds.dtype)
    features = query if features is None else torch.cat([features, query], dim=0)
    if features.shape[0] != num_patch:
      raise ValueError(
        f"Ming-Image patch mismatch: {num_patch} <imagePatch> tokens in the prompt but "
        f"{features.shape[0]} image/query embeddings "
        f"(reference-image blocks come first, then the query block)")
    return embeds.masked_scatter(patch_mask.unsqueeze(-1), features.to(embeds.dtype))

  def prepare_inputs(self,
                     prompt: str,
                     reference_image=None,
                     *,
                     system_prompt: str | None = None) -> MingMllmInputs:
    """Tokenise a prompt, append the query block and compute the 3-axis positions.

    :param prompt: user prompt.
    :param reference_image: optional ``PIL.Image`` used as the editing reference.
    :param system_prompt: overrides the default system sentence.
    :returns: prepared inputs, ready for :meth:`forward`.
    """
    if self.tokenizer is None or self.image_processor is None:
      raise RuntimeError("MingImageMLLM.prepare_inputs needs the tokenizer and image processor")
    text = format_generation_prompt(
      prompt,
      self.tokenizer.eos_token,
      has_reference_image=reference_image is not None,
      system_prompt=system_prompt,
    )
    pixel_values = None
    image_grid_thw = None
    rope_grids: List[List[int]] = []
    if reference_image is not None:
      processed = self.image_processor(images=[reference_image], return_tensors="pt")
      pixel_values = processed["pixel_values"]
      image_grid_thw = processed["image_grid_thw"]
      if image_grid_thw.shape[0] > max(self.max_num_images, 1):
        raise ValueError(
          f"Ming-Image supports at most {self.max_num_images} reference image(s) per prompt")
      text = expand_image_placeholder(text, image_grid_thw[0], self.spatial_merge_size)
      rope_grids.extend(image_grid_thw.tolist())
    input_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if input_ids.ndim == 1:
      input_ids = input_ids.unsqueeze(0)
    prompt_length = int(input_ids.shape[1])
    suffix = torch.tensor([self.query_suffix_ids()], dtype=input_ids.dtype)
    input_ids = torch.cat([input_ids, suffix], dim=1)
    for scale in self.img_gen_scales:
      rope_grids.append([1, 2, scale * scale * 2])
    position_ids = ming_rope_position_ids(
      input_ids,
      torch.tensor(rope_grids, dtype=torch.long),
      image_token_id=self.image_patch_token,
      image_start_token_id=self.image_start_token,
      spatial_merge_size=self.spatial_merge_size,
    )
    return MingMllmInputs(
      input_ids=input_ids,
      attention_mask=torch.ones_like(input_ids),
      position_ids=position_ids,
      image_mask=input_ids == self.image_patch_token,
      pixel_values=pixel_values,
      image_grid_thw=image_grid_thw,
      prompt_length=prompt_length,
      query_start=int(input_ids.shape[1]) - 1 - self.num_query_tokens,
    )

  @staticmethod
  def _causal_keep_mask(attention_mask: torch.Tensor) -> torch.Tensor | None:
    """Build the boolean attention keep-mask.

    :param attention_mask: valid-token mask ``[B, S]``.
    :returns: ``None`` when every token is valid (plain causal attention), else ``[B, 1, S, S]``.
    """
    if bool(attention_mask.all()):
      return None
    seq_len = attention_mask.shape[-1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=attention_mask.device).tril()
    valid = attention_mask.to(torch.bool)[:, None, None, :]
    return causal[None, None] & valid

  def forward(self, inputs: MingMllmInputs) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    """Run the trunk on prepared inputs.

    :param inputs: prepared inputs.
    :returns: final normed states ``[1, S, hidden_size]`` and the captured tensors.
    """
    embeds = self.build_inputs_embeds(inputs)
    attn_mask = self._causal_keep_mask(inputs.attention_mask.to(embeds.device))
    return self.model.model(
      embeds,
      inputs.position_ids.to(embeds.device),
      image_mask=inputs.image_mask.to(embeds.device),
      attn_mask=attn_mask,
      capture_layers=self.capture_layers,
    )

  @torch.no_grad()
  def extract(self,
              prompt: str,
              reference_image=None,
              *,
              system_prompt: str | None = None) -> MingMllmConditions:
    """Extract the raw diffusion conditions of one prompt.

    :param prompt: user prompt.
    :param reference_image: optional reference image for the editing path.
    :param system_prompt: overrides the default system sentence.
    :returns: query hidden states ``[1, num_query_tokens, hidden_size]`` and direct hidden states
      ``[1, prompt_length, len(capture_layers) * hidden_size]``.
    """
    inputs = self.prepare_inputs(prompt, reference_image, system_prompt=system_prompt)
    hidden_states, captured = self.forward(inputs)
    query_hidden_states = hidden_states[:, inputs.query_start:inputs.query_start +
                                        self.num_query_tokens]
    direct_hidden_states = torch.cat(
      [captured[layer][:, :inputs.prompt_length] for layer in self.capture_layers], dim=-1)
    return MingMllmConditions(
      query_hidden_states=query_hidden_states.detach().contiguous(),
      direct_hidden_states=direct_hidden_states.detach().contiguous(),
    )

  @torch.no_grad()
  def encode(self,
             prompt: str,
             image=None,
             *,
             system_prompt: str | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the conditions as the ``(query, direct)`` pair the pipeline consumes.

    :param prompt: user prompt.
    :param image: optional reference image for the editing path.
    :param system_prompt: overrides the default system sentence.
    :returns: ``(query_hidden_states, direct_hidden_states)``.
    """
    conditions = self.extract(prompt, image, system_prompt=system_prompt)
    return conditions.query_hidden_states, conditions.direct_hidden_states


__all__ = [
  "MingBailingForCausalLM",
  "MingBailingModel",
  "MingImageMLLM",
  "MingMllmConditions",
  "MingMllmInputs",
  "MingMoeBlock",
  "MingRouter",
  "apply_3d_rotary_pos_emb",
  "expand_image_placeholder",
  "format_generation_prompt",
  "iter_checkpoint_tensors",
  "load_ming_mllm_weights",
  "load_query_tokens",
  "ming_rope_position_ids",
]
