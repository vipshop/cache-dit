"""Context-parallel plan for LTX-2 video transformers.

This planner reuses the LTXVideo implementation pattern, but adjusts the plan and patched helper
logic for LTX-2-specific timestep and mask behavior.
"""

import functools
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_ltx2 import (
  LTX2Attention,
  LTX2AudioVideoAttnProcessor,
  LTX2PerturbedAttnProcessor,
  LTX2VideoTransformer3DModel,
  apply_interleaved_rotary_emb,
  apply_split_rotary_emb,
)
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

# Upstream `__call__` implementations captured before the class-level CP patches are installed.
# The patched `__call__` hands control back to them for any processor without a multi-rank
# `_cp_config` (e.g. the `LTX2TextConnectors` component), so the patch can never rewrite non-CP math.
_CP_PATCH_ORIGINALS: dict = {}


def _cp_is_armed(processor) -> bool:
  """Whether `processor` was armed by the CP runtime with a multi-rank `_cp_config`."""
  cp_config = getattr(processor, "_cp_config", None)
  return cp_config is not None and cp_config._world_size > 1


def _install_cp_processor_patches() -> None:
  """Install the class-level attention patches the LTX-2 CP path needs.

  Both processors must be patched: LTX-2.5 checkpoints set `perturbed_attn=True`, and
  `LTX2PerturbedAttnProcessor` is a standalone class (not a subclass of
  `LTX2AudioVideoAttnProcessor`). Patching only the latter silently disables CP for LTX-2.5: the
  sequence is already split across ranks, so attention would run on local shards only.

  The patches are class-level, so they also cover attention trees the CP runtime never armed with
  `_cp_config` -- most importantly `LTX2TextConnectors`, which builds its attention from
  `LTX2Attention` + `LTX2AudioVideoAttnProcessor` with `gated_attn=True` on LTX-2.5. The patched
  `__call__` therefore keeps the untouched upstream implementation and only takes over when the
  processor actually carries a multi-rank `_cp_config`; anything else must stay bit-identical to a
  cp-free run.
  """
  for processor_cls in (LTX2AudioVideoAttnProcessor, LTX2PerturbedAttnProcessor):
    _CP_PATCH_ORIGINALS.setdefault(processor_cls, processor_cls.__call__)
  LTX2Attention.prepare_attention_mask = __patch__LTX2Attention_prepare_attention_mask__  # type: ignore[assignment]
  LTX2AudioVideoAttnProcessor.__call__ = __patch__LTX2AudioVideoAttnProcessor__call__  # type: ignore[assignment]
  LTX2PerturbedAttnProcessor.__call__ = __patch__LTX2PerturbedAttnProcessor__call__  # type: ignore[assignment]


@ContextParallelismPlannerRegister.register("LTX2")
class LTX2ContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert isinstance(
      transformer,
      LTX2VideoTransformer3DModel), "Transformer must be an instance of LTX2VideoTransformer3DModel"

    # LTX2ImageToVideoPipeline passes `timestep` as a 2D `(B, seq_len)` tensor named
    # `video_timestep`. The cache-dit CP plan shards that tensor correctly for Ulysses / Ring CP.

    # Patch attention-mask preparation and both processors so head sharding, global-sequence
    # padding and the CP dispatch stay aligned under CP. Unarmed processors (e.g. the
    # `LTX2TextConnectors` component) hand control back to the upstream `__call__`.
    _install_cp_processor_patches()

    rope_type = getattr(getattr(transformer, "config", None), "rope_type", "interleaved")
    if rope_type == "split":
      # split RoPE returns (B, H, T, D/2), shard along T dim
      rope_expected_dims = 4
      rope_split_dim = 2
    else:
      # interleaved RoPE returns (B, T, D), shard along T dim
      rope_expected_dims = 3
      rope_split_dim = 1

    _cp_plan: _ContextParallelModelPlan = {
      "": {
        # Shard video/audio latents across sequence
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "audio_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        # Shard prompt embeds across sequence
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "audio_encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        # IMPORTANT: shard video timestep (B, seq_len) to match sharded hidden_states
        "timestep":
        _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        # Echo (JoyAI-Echo) drives the audio stream with a per-token audio timestep (B, seq_len_a),
        # which has to follow the sharded `audio_hidden_states`. LTX-2 / LTX-2.5 pass a scalar
        # `(B,)` instead: `expected_dims=2` makes the hook skip it with a `warning_once` and the
        # value stays replicated, which is what those pipelines want (it broadcasts per token).
        "audio_timestep":
        _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        # LTX-2.5 keyframe marker (B, seq_len, 1). It never reaches the blocks, but the transformer
        # adds it onto the already-split `hidden_states` right after `proj_in`, so it has to be split
        # on the same axis. `video_coords` (B, 3, seq_len, 2) is different: it is consumed by `rope`,
        # whose output is split below, so the coordinates themselves stay full-length.
        "video_keyframes_mask":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        # NOTE: do NOT shard attention masks; handled in patched attention processor
      },
      # Split RoPE outputs to match CP-sharded sequence length
      "rope": {
        0:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
        1:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
      },
      "audio_rope": {
        0:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
        1:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
      },
      "cross_attn_rope": {
        0:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
        1:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
      },
      "cross_attn_audio_rope": {
        0:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
        1:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
      },
      # Gather outputs before returning
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      "audio_proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    return _cp_plan


# Upstream references for future syncs:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx2.py
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py


@functools.wraps(LTX2Attention.prepare_attention_mask)
def __patch__LTX2Attention_prepare_attention_mask__(
  self: LTX2Attention,
  attention_mask: torch.Tensor,
  target_length: int,
  batch_size: int,
  out_dim: int = 3,
  # NOTE: Allow specifying head_size for CP
  head_size: Optional[int] = None,
) -> torch.Tensor:
  # Relative to upstream diffusers, this helper accepts an explicit `head_size`. Under Context
  # Parallelism each rank owns only `attn.heads // world_size` heads, so repeating the mask with the
  # full `self.heads` would break the sharded attention shape contract.
  if head_size is None:
    head_size = self.heads
  if attention_mask is None:
    return attention_mask

  current_length: int = attention_mask.shape[-1]
  if current_length != target_length:
    if attention_mask.device.type == "mps":
      padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
      padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
      attention_mask = torch.cat([attention_mask, padding], dim=2)
    else:
      attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

  if out_dim == 3:
    if attention_mask.shape[0] < batch_size * head_size:
      attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
  elif out_dim == 4:
    attention_mask = attention_mask.unsqueeze(1)
    attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

  return attention_mask


def _prepare_cp_attn_mask(
  processor: object,
  attn: "LTX2Attention",
  attention_mask: Optional[torch.Tensor],
  sequence_length: int,
  batch_size: int,
) -> Optional[torch.Tensor]:
  """Expand a text attention mask to the global sequence / local head layout used under CP.

  Diffusers prepares the mask from the local `sequence_length` and reshapes it with the full
  `attn.heads`. Under Context Parallelism the sequence is sharded across ranks while the mask still
  represents the global sequence, so `target_length` is expanded back to the global sequence length
  and the mask is repeated with `attn.heads // world_size`.

  :param processor: Attention processor, optionally carrying `_cp_config`.
  :param attn: The `LTX2Attention` module the mask belongs to.
  :param attention_mask: Additive text mask, `(batch, 1, text_len)`.
  :param sequence_length: Local sequence length of the masked stream.
  :param batch_size: Batch size of the masked stream.
  :returns: The reshaped mask, or `None` when no mask was given.
  """
  if attention_mask is None:
    return None

  # The class-level patch also covers processors the CP runtime never visited (e.g. pipelines that
  # hold a second attention tree), which simply take the plain path.
  cp_config = getattr(processor, "_cp_config", None)
  if cp_config is None or cp_config._world_size <= 1:
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    return attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

  head_size = attn.heads // cp_config._world_size
  attention_mask = attn.prepare_attention_mask(
    attention_mask,
    sequence_length * cp_config._world_size,
    batch_size,
    3,
    head_size,
  )
  return attention_mask.view(batch_size, head_size, -1, attention_mask.shape[-1])


@functools.wraps(LTX2AudioVideoAttnProcessor.__call__)
def __patch__LTX2AudioVideoAttnProcessor__call__(
  self: LTX2AudioVideoAttnProcessor,
  attn: "LTX2Attention",
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  query_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
  key_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
  original = _CP_PATCH_ORIGINALS.get(LTX2AudioVideoAttnProcessor)
  if original is not None and not _cp_is_armed(self):
    return original(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      query_rotary_emb=query_rotary_emb,
      key_rotary_emb=key_rotary_emb,
    )

  batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else
                                    encoder_hidden_states.shape)

  attention_mask = _prepare_cp_attn_mask(self, attn, attention_mask, sequence_length, batch_size)

  if encoder_hidden_states is None:
    encoder_hidden_states = hidden_states

  # Gated attention (`gated_attn=True`, LTX-2.3/2.5 and the LTX2TextConnectors component). The
  # class-level patch also reaches processors outside the CP-enabled transformer -- the connectors
  # build their attention from this very class -- so dropping the gate silently rewrote the prompt
  # embeddings under CP and produced a different (but coherent) video.
  gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None

  query = attn.to_q(hidden_states)
  key = attn.to_k(encoder_hidden_states)
  value = attn.to_v(encoder_hidden_states)

  query = attn.norm_q(query)
  key = attn.norm_k(key)

  if query_rotary_emb is not None:
    # Keep RoPE logic identical to upstream: for v2a/a2v cross-attn, K can use separate RoPE.
    if attn.rope_type == "interleaved":
      query = apply_interleaved_rotary_emb(query, query_rotary_emb)
      key = apply_interleaved_rotary_emb(
        key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)
    elif attn.rope_type == "split":
      query = apply_split_rotary_emb(query, query_rotary_emb)
      key = apply_split_rotary_emb(
        key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)

  query = query.unflatten(2, (attn.heads, -1))
  key = key.unflatten(2, (attn.heads, -1))
  value = value.unflatten(2, (attn.heads, -1))

  hidden_states = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=self._attention_backend,
    cp_config=getattr(self, "_cp_config", None),
  )
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.to(query.dtype)

  if gate_logits is not None:
    hidden_states = hidden_states.unflatten(2, (attn.heads, -1))  # [B, T, H, D]
    # The factor of 2.0 keeps the initial gates at 1 for zero-initialized gate logits. `gate_logits`
    # was computed from the CP-local `hidden_states`, so its token axis already matches.
    gates = 2.0 * torch.sigmoid(gate_logits)  # [B, T, H]
    hidden_states = hidden_states * gates.unsqueeze(-1)
    hidden_states = hidden_states.flatten(2, 3)

  hidden_states = attn.to_out[0](hidden_states)
  hidden_states = attn.to_out[1](hidden_states)
  return hidden_states


@functools.wraps(LTX2PerturbedAttnProcessor.__call__)
def __patch__LTX2PerturbedAttnProcessor__call__(
  self: LTX2PerturbedAttnProcessor,
  attn: "LTX2Attention",
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  query_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
  key_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
  perturbation_mask: Optional[torch.Tensor] = None,
  all_perturbed: Optional[bool] = None,
) -> torch.Tensor:
  # CP counterpart of `LTX2PerturbedAttnProcessor.__call__`. Only the attention dispatch and the
  # mask preparation change; the spatio-temporal-guidance lerp and the per-head gate stay identical
  # to upstream, because both operate on per-token tensors that CP already sharded (Ulysses/Ring
  # return `(B, seq_local, heads_global, dim)`), so no cross-rank reduction is needed for them.
  original = _CP_PATCH_ORIGINALS.get(LTX2PerturbedAttnProcessor)
  if original is not None and not _cp_is_armed(self):
    return original(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      query_rotary_emb=query_rotary_emb,
      key_rotary_emb=key_rotary_emb,
      perturbation_mask=perturbation_mask,
      all_perturbed=all_perturbed,
    )

  batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else
                                    encoder_hidden_states.shape)

  attention_mask = _prepare_cp_attn_mask(self, attn, attention_mask, sequence_length, batch_size)

  if encoder_hidden_states is None:
    encoder_hidden_states = hidden_states

  gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None

  value = attn.to_v(encoder_hidden_states)
  if all_perturbed is None:
    all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False

  if all_perturbed:
    # Skip attention, use the value projection value.
    hidden_states = value
  else:
    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if query_rotary_emb is not None:
      if attn.rope_type == "interleaved":
        query = apply_interleaved_rotary_emb(query, query_rotary_emb)
        key = apply_interleaved_rotary_emb(
          key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)
      elif attn.rope_type == "split":
        query = apply_split_rotary_emb(query, query_rotary_emb)
        key = apply_split_rotary_emb(
          key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    hidden_states = _dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=attention_mask,
      dropout_p=0.0,
      is_causal=False,
      backend=self._attention_backend,
      cp_config=getattr(self, "_cp_config", None),
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    if perturbation_mask is not None:
      # `value` was projected from the same CP-sharded stream, so its token axis matches the
      # dispatched output. Assert it instead of relying on the implicit CP output contract.
      value = value.flatten(2, 3)
      assert value.shape[1] == hidden_states.shape[1], (
        f"CP output token count {hidden_states.shape[1]} does not match the value projection "
        f"{value.shape[1]}; the attention backend no longer returns local-sequence outputs.")
      hidden_states = torch.lerp(value, hidden_states, perturbation_mask)

  if gate_logits is not None:
    hidden_states = hidden_states.unflatten(2, (attn.heads, -1))
    # The factor of 2.0 keeps the initial gates at 1 for zero-initialized gate logits.
    gates = 2.0 * torch.sigmoid(gate_logits)
    hidden_states = hidden_states * gates.unsqueeze(-1)
    hidden_states = hidden_states.flatten(2, 3)

  hidden_states = attn.to_out[0](hidden_states)
  hidden_states = attn.to_out[1](hidden_states)
  return hidden_states
