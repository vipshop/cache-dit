"""FastH3: the few-step DMD2 students of MiniMax-H3, with video sparse attention.

`FastVideo/FastVideo-FastH3-*` releases are MiniMax-H3 checkpoints distilled to 4 or 8 transformer
evaluations. They differ from the base model in exactly three ways, and all three live here:

* **A VSA compression gate per block.** Every `transformer_blocks[i].attn` gains a
  `to_gate_compress` linear (`hidden -> heads * head_dim`, no bias). The base checkpoint has no such
  parameter, so a plain `from_pretrained` silently drops the 50 tensors (diffusers only logs
  unexpected keys) and the attention degrades to dense — which is a different model.
* **A fixed DMD sigma ladder.** The repository ships `fastvideo_inference.json` with the rungs, the
  forward count and the attention recipe; the per-modality shift comes from the two scheduler
  configs. `set_timesteps(sigmas=...)` uses the values verbatim, so the shift has to be applied
  here, once per modality.
* **Video sparse attention.** Video rows attend to the dense prefix plus the top-k 3-D tiles of
  video keys, and the gate adds a compression branch over the pooled tile means.

The class stays a `MiniMaxH3Transformer3DModel` subclass so cache-dit's dispatch keeps working:

Note:
  Keep the constraints below intact, cache-dit's dispatch depends on them:

  * The class must stay a `MiniMaxH3Transformer3DModel` subclass with the parent's module tree
    (`transformer_blocks`, `token_refiner`, `norm_out`, `proj_in`, `proj_out`, ...), so
    `ForwardPattern.Pattern_3`, the MiniMax-H3 context/tensor parallel planners and the layerwise
    offload groups keep resolving.
  * `_repeated_blocks` and `_keep_in_fp32_modules` are inherited: regional quantization and
    `compile_repeated_blocks()` find the denoising stack, and the float32 projections keep their
    dtype.
  * `forward` is **not** overridden. Its signature is what the modular pipeline inspects to decide
    which layout tensors to pass, and it carries the LoRA scale decorator; the VSA geometry is read
    from a forward pre-hook instead.
  * `to_gate_compress` is created in `__init__`, i.e. before `from_pretrained` walks the state
    dict, so the 50 gate tensors are consumed rather than reported as unexpected keys. For a
    checkpoint without gates (`-Dense-DataFree`) the parameter is not built at all: the model loads
    with `low_cpu_mem_usage` under `init_empty_weights`, so a parameter that no checkpoint shard
    fills would stay on the meta device and crash on the first `.to(device)`.
  * Only the transformer is owned here. The text encoder, both VAEs and the schedulers stay
    diffusers classes, because cache-dit detects encoders by module path (`_is_auto_encoder`,
    `_is_text_encoder`); vendoring them would silently disable `--parallel-vae` / `--parallel-text`.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Any, Optional

import torch
from diffusers.configuration_utils import register_to_config
from diffusers.models.transformers.transformer_minimax_h3 import (
  MiniMaxH3Transformer3DModel,
  _apply_rotary_emb,
)
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline
from torch import nn

from ...attention.backends.register import _dispatch_attention_fn
from ...attention.vsa_h3 import vsa_h3_attention
from ...distributed.core._distributed_primitives import (
  _all_to_all_single_o_custom_heads,
  _all_to_all_single_qkv_custom_heads,
)
from ...attention.vsa_h3_geometry import VsaH3Meta, vsa_h3_meta_from_layout
from ...logger import init_logger

logger = init_logger(__name__)

__all__ = [
  "FastH3AttnProcessor",
  "FastH3Contract",
  "FastH3ModularPipeline",
  "FastH3Transformer3DModel",
  "FASTH3_INFERENCE_FILE",
]

FASTH3_INFERENCE_FILE = "fastvideo_inference.json"
FASTH3_TILE_SHAPE = (4, 4, 4)
FASTH3_TILE_ELEMENTS = 64


@dataclasses.dataclass(frozen=True)
class FastH3Contract:
  """Sampling contract of one FastH3 release, as published next to its weights.

  :param num_inference_steps: sigma grid points; the transformer runs one fewer forward.
  :param transformer_forwards: number of transformer evaluations the student was distilled for.
  :param dmd_denoising_steps: DMD rungs on the 1000-step training clock, before the shift.
  :param guidance_scale: guidance the student expects; FastH3 is guidance-distilled, i.e. ``1.0``.
  :param vsa_tile_size: VSA tile volume in tokens, ``64`` for every shipped student.
  :param vsa_sparsity: fraction of video tiles the attention drops.
  :param video_shift: exponential sigma shift of the video scheduler.
  :param audio_shift: exponential sigma shift of the audio scheduler.
  :param task: task the student was distilled for; only ``t2av`` exists.
  """

  num_inference_steps: int
  transformer_forwards: int
  dmd_denoising_steps: tuple[int, ...]
  guidance_scale: float
  vsa_tile_size: int
  vsa_sparsity: float
  video_shift: float
  audio_shift: float
  task: str

  @property
  def requires_vsa(self) -> bool:
    """Whether the release was distilled with the sparse recipe (all but `-Dense-DataFree`)."""
    return self.vsa_tile_size > 0 and self.vsa_sparsity > 0.0

  def sigmas(self, shift: float, device: torch.device | str = "cpu") -> torch.Tensor:
    """Shifted sigma grid of one modality, terminal zero included.

    The rungs are noise levels on the 1000-step clock, so they are divided first and shifted once
    with ``sigma' = shift * sigma / (1 + (shift - 1) * sigma)``; the scheduler consumes the result
    verbatim.

    :param shift: exponential shift of the modality's scheduler.
    :param device: device of the returned tensor.
    :returns: sigma grid, shape ``(num_inference_steps,)``.
    """
    base = torch.tensor([step / 1000.0 for step in self.dmd_denoising_steps] + [0.0],
                        dtype=torch.float32,
                        device=device)
    return float(shift) * base / (1.0 + (float(shift) - 1.0) * base)

  @classmethod
  def from_checkpoint(cls, path: str | os.PathLike) -> Optional["FastH3Contract"]:
    """Read the contract of a FastH3 repository, or ``None`` for a base MiniMax-H3 checkpoint.

    :param path: checkpoint directory.
    :returns: The contract, or ``None`` when the repository does not ship one.
    """
    contract_path = os.path.join(str(path), FASTH3_INFERENCE_FILE)
    if not os.path.isfile(contract_path):
      return None
    with open(contract_path, encoding="utf-8") as handle:
      payload = json.load(handle)

    forwards = int(payload.get("transformer_forwards", 0))
    steps = payload.get("dmd_denoising_steps") or []
    if forwards < 1 or len(steps) != forwards:
      raise ValueError(f"{contract_path} is inconsistent: transformer_forwards={forwards} but "
                       f"dmd_denoising_steps={steps}.")
    ladder = [int(step) for step in steps]
    if any(step < 0 or step > 1000 for step in ladder):
      raise ValueError(f"{contract_path} has rungs outside the 0..1000 training clock: {ladder}.")
    if any(earlier <= later for earlier, later in zip(ladder, ladder[1:])):
      raise ValueError(f"{contract_path} rungs must be strictly decreasing (noise level order): "
                       f"{ladder}.")
    grid_points = int(payload.get("num_inference_steps", forwards + 1))
    if grid_points != forwards + 1:
      raise ValueError(f"{contract_path} declares {grid_points} sigma grid points for {forwards} "
                       "transformer forward passes; the scheduler needs exactly one more point "
                       "than forward passes.")
    tile_size = int(payload.get("vsa_tile_size", FASTH3_TILE_ELEMENTS))
    if tile_size not in (0, FASTH3_TILE_ELEMENTS):
      raise ValueError(f"{contract_path} asks for {tile_size}-token VSA tiles; the FastH3 students "
                       f"are distilled with {FASTH3_TILE_ELEMENTS}-token tiles.")
    return cls(
      num_inference_steps=grid_points,
      transformer_forwards=forwards,
      dmd_denoising_steps=tuple(ladder),
      guidance_scale=float(payload.get("guidance_scale", 1.0)),
      vsa_tile_size=tile_size,
      vsa_sparsity=float(payload.get("vsa_sparsity", 0.0)),
      video_shift=_resolve_shift(path, payload, "scheduler", "video_scheduler_shift"),
      audio_shift=_resolve_shift(path, payload, "audio_scheduler", "audio_scheduler_shift"),
      task=str(payload.get("task", "t2av")),
    )


def _resolve_shift(
  path: str | os.PathLike,
  payload: dict,
  subfolder: str,
  contract_key: str,
) -> float:
  """Pick the modality shift, preferring the contract over the scheduler config.

  The contract is the distillation record, so it wins when it states a shift (the 8-step release
  carries `video_scheduler_shift` / `audio_scheduler_shift`; the 4-step preview only ships the two
  scheduler configs). A config that disagrees is reported, because that is exactly the kind of
  stale-file drift that would silently resample the student off its ladder.

  :param path: checkpoint directory.
  :param payload: parsed `fastvideo_inference.json`.
  :param subfolder: scheduler subfolder in the repository.
  :param contract_key: key that may carry the shift in the contract.
  :returns: The shift to apply to that modality's sigma grid.
  """
  config_path = os.path.join(str(path), subfolder, "scheduler_config.json")
  config_shift = None
  if os.path.isfile(config_path):
    with open(config_path, encoding="utf-8") as handle:
      config_shift = float(json.load(handle).get("shift", 1.0))
  contract_shift = payload.get(contract_key)
  if contract_shift is None:
    if config_shift is None:
      raise ValueError(f"FastH3 needs either `{contract_key}` in the contract or `{config_path}` "
                       "to build the sigma grid.")
    return config_shift
  contract_shift = float(contract_shift)
  if config_shift is not None and abs(config_shift - contract_shift) > 1e-6:
    logger.warning(f"FastH3 {subfolder}: the contract says shift={contract_shift} but "
                   f"{config_path} says shift={config_shift}; using the contract.")
  return contract_shift


def _transformer_mounts_vsa_gates(path: str | os.PathLike, subfolder: str = "transformer") -> bool:
  """Whether the transformer shards carry `attn.to_gate_compress` weights."""
  root = os.path.join(str(path), subfolder)
  index_path = os.path.join(root, "diffusion_pytorch_model.safetensors.index.json")
  if os.path.isfile(index_path):
    with open(index_path, encoding="utf-8") as handle:
      keys = json.load(handle).get("weight_map", {})
    return any("to_gate_compress" in key for key in keys)
  if not os.path.isdir(root):
    raise ValueError(f"FastH3 expects the transformer shards under `{root}`, which does not exist.")
  from safetensors import safe_open

  shards = sorted(name for name in os.listdir(root) if name.endswith(".safetensors"))
  if not shards:
    raise ValueError(f"FastH3 found no `.safetensors` shards in `{root}`; the download may be "
                     "incomplete.")
  for name in shards:
    with safe_open(os.path.join(root, name), framework="pt") as handle:
      if any("to_gate_compress" in key for key in handle.keys()):
        return True
  return False


def _is_dtensor(tensor: Any) -> bool:
  from torch.distributed.tensor import DTensor

  return isinstance(tensor, DTensor)


def _to_local(tensor: torch.Tensor) -> torch.Tensor:
  """Unwrap a DTensor to its local shard; a plain tensor passes through."""
  return tensor.to_local() if _is_dtensor(tensor) else tensor


def _to_local_tensors(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
  """Move Q/K/V out of the DTensor world for the sparse kernel.

  Tile selection calls `index_select` / `gather` / `topk` and then hands a local block mask to
  FlexAttention; none of those has a DTensor strategy, and the failure would surface inside
  dispatch. The attention is head-wise, so the local shard is the whole computation for this rank.

  :param query: query tensor, possibly a DTensor sharded on the head axis.
  :param key: key tensor.
  :param value: value tensor.
  :returns: Local Q/K/V and the tensor whose placement the output must be rewrapped with.
  """
  reference = next((t for t in (query, key, value) if _is_dtensor(t)), None)
  return _to_local(query), _to_local(key), _to_local(value), reference


def _restore_placement(local: torch.Tensor, reference: Optional[torch.Tensor]) -> torch.Tensor:
  """Wrap a local attention output back into the placement its inputs had."""
  if reference is None:
    return local
  from torch.distributed.tensor import DTensor

  return DTensor.from_local(local, reference.device_mesh, reference.placements, run_check=False)


def _ulysses_world(parallel_config: Any) -> int:
  return int(getattr(parallel_config, "ulysses_degree", 1) or 1)


def _ulysses_group(parallel_config: Any):
  mesh = getattr(parallel_config, "_ulysses_mesh", None)
  if mesh is None:
    raise ValueError("FastH3 video sparse attention needs a Ulysses mesh for context parallelism.")
  return mesh.get_group()


@torch._dynamo.disable
def _ulysses_gather_sequence(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  gate: Optional[torch.Tensor],
  heads: int,
  parallel_config: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], list[int], int]:
  """Turn ``(B, S_local, H, D)`` row shards into ``(B, S_global, H_local, D)``.

  Ulysses scatters heads and gathers the sequence, so after this call every rank holds the whole
  packed sequence and is therefore able to rank tiles exactly like the single-GPU path. The
  compression gate rides the same all-to-all — it is per head and per channel, so it must land in
  the same layout as Q/K/V (this is what FastVideo and vllm-omni do; sglang leaves the gate in the
  row shard and folds after the output exchange instead).

  :param query: local query shard, shape ``(B, S_local, H_global, D)``.
  :param key: local key shard.
  :param value: local value shard.
  :param gate: optional per-head compression gate in the same layout as ``query``.
  :param heads: global attention head count.
  :param parallel_config: cache-dit context-parallel config.
  This is a graph break on purpose: `all_to_all_single` is a collective and cannot run under
  Dynamo's fake-tensor tracing (`RuntimeError when making fake tensor call`). Keeping the exchange
  opaque is also what FastVideo/xDiT do with their collectives, and it leaves the sparse attention
  itself fully compilable.

  :returns: Communicated Q/K/V/gate, the per-rank head split sizes and the local row count.
  """
  world = _ulysses_world(parallel_config)
  if heads % world:
    raise ValueError(f"FastH3 VSA needs the head count ({heads}) to divide the Ulysses degree "
                     f"({world}); 56 heads allow {[d for d in range(1, 57) if 56 % d == 0]}.")
  group = _ulysses_group(parallel_config)
  split_sizes = [heads // world] * world
  local_rows = int(query.shape[1])
  query = _all_to_all_single_qkv_custom_heads(query, group, split_sizes)
  key = _all_to_all_single_qkv_custom_heads(key, group, split_sizes)
  value = _all_to_all_single_qkv_custom_heads(value, group, split_sizes)
  if gate is not None:
    gate = _all_to_all_single_qkv_custom_heads(gate, group, split_sizes)
  return query, key, value, gate, split_sizes, local_rows


@torch._dynamo.disable
def _ulysses_split_sequence(
  out: torch.Tensor,
  split_sizes: list[int],
  local_rows: int,
  parallel_config: Any,
) -> torch.Tensor:
  """Undo :func:`_ulysses_gather_sequence`: heads back to global, sequence back to rows."""
  return _all_to_all_single_o_custom_heads(out, _ulysses_group(parallel_config), split_sizes,
                                           local_rows)


class FastH3AttnProcessor:
  """MiniMax-H3 attention with FastH3's video sparse attention on top.

  Mirrors `MiniMaxH3AttnProcessor` up to the attention call, then routes to
  :func:`cache_dit.attention.vsa_h3.vsa_h3_attention` when the request's tile geometry is
  available and to the configured dense backend otherwise.
  """

  _attention_backend = None
  _parallel_config = None

  def __call__(
    self,
    attn: nn.Module,
    hidden_states: torch.Tensor,
    rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    if getattr(attn, "fused_projections", False):
      query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
    else:
      query = attn.to_q(hidden_states)
      key = attn.to_k(hidden_states)
      value = attn.to_v(hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    if rotary_emb is not None:
      query = _apply_rotary_emb(query, *rotary_emb)
      key = _apply_rotary_emb(key, *rotary_emb)

    meta = getattr(attn, "_vsa_meta", None)
    gate = None
    if meta is not None and not getattr(attn, "_vsa_gate_active", False):
      # No trained gate: this checkpoint was not distilled with VSA (base MiniMax-H3 or
      # `-Dense-DataFree`), so the faithful path is plain dense attention.
      meta = None
    if meta is not None:
      if attention_mask is not None:
        raise ValueError(
          "FastH3 video sparse attention does not accept an attention mask: the packed sequence "
          "carries no padding and the tile geometry supplies the visibility.")
      gate = attn.to_gate_compress(hidden_states).unflatten(-1, (attn.heads, -1))
      parallel_config = getattr(self, "_parallel_config", None)
      if parallel_config is not None and getattr(parallel_config, "ulysses_degree", 1) > 1:
        # Ulysses gathers the packed sequence back on every rank (heads are split instead), which
        # is exactly the layout tile selection needs; the compression gate rides the same
        # all-to-all so it stays aligned with Q/K/V (FastVideo / vllm-omni do the same).
        query, key, value, gate, split_sizes, local_rows = _ulysses_gather_sequence(
          query, key, value, gate, attn.heads, parallel_config)
        hidden_states = vsa_h3_attention(query,
                                         key,
                                         value,
                                         meta,
                                         gate=gate,
                                         sparsity=getattr(attn, "_vsa_sparsity", meta.sparsity))
        hidden_states = _ulysses_split_sequence(hidden_states, split_sizes, local_rows,
                                                parallel_config)
      else:
        query, key, value, hidden_ref = _to_local_tensors(query, key, value)
        gate_local = _to_local(gate)
        hidden_states = vsa_h3_attention(query,
                                         key,
                                         value,
                                         meta,
                                         gate=gate_local,
                                         sparsity=getattr(attn, "_vsa_sparsity", meta.sparsity))
        hidden_states = _restore_placement(hidden_states, hidden_ref)
    else:
      hidden_states = _dispatch_attention_fn(query,
                                             key,
                                             value,
                                             attn_mask=attention_mask,
                                             dropout_p=0.0,
                                             is_causal=False,
                                             backend=self._attention_backend,
                                             parallel_config=self._parallel_config)
    hidden_states = hidden_states.flatten(2, 3).type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


class FastH3Transformer3DModel(MiniMaxH3Transformer3DModel):
  """MiniMax-H3 transformer with FastH3's compression gates and sparse attention."""

  # The parent's parameters are repeated instead of absorbed by `**kwargs`: diffusers builds a
  # model with `extract_init_dict`, which keeps only the config keys that appear in
  # `inspect.signature(cls.__init__)`. A catch-all signature would silently drop the checkpoint's
  # architecture and build the default 56-head / 50-layer model instead.
  @register_to_config
  def __init__(
    self,
    num_attention_heads: int = 56,
    attention_head_dim: int = 128,
    hidden_size: int = 5376,
    num_layers: int = 50,
    num_refiner_layers: int = 2,
    ffn_dim: int = 14336,
    in_channels: int = 24,
    audio_in_channels: int = 32,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    text_dim: int = 5120,
    freq_dim: int = 256,
    time_embed_hidden_dim: int = 5376,
    time_embed_dim: int = 2688,
    rope_freq_dim: int = 16,
    rope_theta: float = 10000.0,
    norm_eps: float = 1e-5,
    qk_norm_eps: float = 1e-5,
    final_norm_eps: float = 1e-5,
    enable_vsa: bool = True,
    vsa_sparsity: float = 0.9,
  ) -> None:
    super().__init__(
      num_attention_heads=num_attention_heads,
      attention_head_dim=attention_head_dim,
      hidden_size=hidden_size,
      num_layers=num_layers,
      num_refiner_layers=num_refiner_layers,
      ffn_dim=ffn_dim,
      in_channels=in_channels,
      audio_in_channels=audio_in_channels,
      patch_size=patch_size,
      text_dim=text_dim,
      freq_dim=freq_dim,
      time_embed_hidden_dim=time_embed_hidden_dim,
      time_embed_dim=time_embed_dim,
      rope_freq_dim=rope_freq_dim,
      rope_theta=rope_theta,
      norm_eps=norm_eps,
      qk_norm_eps=qk_norm_eps,
      final_norm_eps=final_norm_eps,
    )
    self.enable_vsa = bool(enable_vsa)
    self.vsa_sparsity = float(vsa_sparsity)
    self._vsa_meta_cache: dict[tuple, VsaH3Meta] = {}
    if self.enable_vsa:
      self._install_vsa_attention()
    self.register_forward_pre_hook(self._vsa_layout_hook, with_kwargs=True)

  def _install_vsa_attention(self) -> None:
    heads = self.config.num_attention_heads
    head_dim = self.config.attention_head_dim
    for block in self.transformer_blocks:
      attn = block.attn
      attn.to_gate_compress = nn.Linear(self.config.hidden_size, heads * head_dim, bias=False)
      # Zero init matches the reference: an unfilled gate contributes exactly nothing, so a
      # checkpoint without gates (`-Dense-DataFree`, or the base MiniMax-H3) runs dense instead
      # of silently mixing in a random projection.
      nn.init.zeros_(attn.to_gate_compress.weight)
      attn._vsa_gate_active = None
      attn._vsa_sparsity = self.vsa_sparsity
      attn._vsa_meta = None
      attn.set_processor(FastH3AttnProcessor())

  def refresh_vsa_gate_state(self) -> None:
    """Decide once whether the compression branch contributes, after the weights are loaded.

    The branch is a `Linear` the checkpoint may fill with zeros (base MiniMax-H3 has no gate at
    all); a structurally zero gate makes the whole branch a guaranteed zero, so it is skipped
    instead of running one extra GEMM plus a wider all-to-all per layer.
    """
    if not self.enable_vsa:
      return
    for block in self.transformer_blocks:
      attn = block.attn
      weight = getattr(attn, "to_gate_compress", None)
      attn._vsa_gate_active = weight is not None and bool(torch.count_nonzero(weight.weight).item())
    self._vsa_state_resolved = True
    active = sum(1 for block in self.transformer_blocks if block.attn._vsa_gate_active)
    logger.info(f"FastH3 VSA gates: {active}/{len(self.transformer_blocks)} blocks carry a "
                "trained compression gate.")

  def _vsa_layout_hook(self, module: nn.Module, args: tuple, kwargs: dict) -> None:
    if not self.enable_vsa:
      return None
    if not getattr(self, "_vsa_state_resolved", False):
      # One host sync, on the first step, whichever entry point built the model: loading through
      # the pipeline resolves it right after `load_components`, a bare
      # `FastH3Transformer3DModel.from_pretrained` resolves it here.
      self.refresh_vsa_gate_state()
    position_ids = kwargs.get("position_ids")
    video_indices = kwargs.get("video_indices")
    audio_indices = kwargs.get("audio_indices")
    text_indices = kwargs.get("text_indices")
    if position_ids is None and args:
      # The modular denoise step passes the layout as keywords, but a caller may bind them
      # positionally; resolve by name so the sparse path does not silently degrade to dense.
      import inspect

      bound = inspect.signature(self.forward).bind_partial(self, *args, **kwargs)
      position_ids = bound.arguments.get("position_ids")
      video_indices = bound.arguments.get("video_indices")
      audio_indices = bound.arguments.get("audio_indices")
      text_indices = bound.arguments.get("text_indices")
    if position_ids is None or video_indices is None or audio_indices is None or text_indices is None:
      return None
    key = (
      int(position_ids.shape[0]),
      int(video_indices.numel()),
      int(audio_indices.numel()),
      int(text_indices.numel()),
      str(position_ids.device),
    )
    meta = self._vsa_meta_cache.get(key)
    if meta is None:
      meta = vsa_h3_meta_from_layout(position_ids,
                                     video_indices,
                                     audio_indices,
                                     text_indices,
                                     sparsity=self.vsa_sparsity,
                                     tile_shape=FASTH3_TILE_SHAPE)
      self._vsa_meta_cache[key] = meta
      logger.debug(f"FastH3 VSA geometry: {meta.num_tiles} tiles "
                   f"({meta.num_prefix_tiles} dense prefix), {meta.num_video_tiles} video.")
    for block in self.transformer_blocks:
      block.attn._vsa_meta = meta
    return None

  def set_vsa_enabled(self, enabled: bool) -> None:
    """Turn the sparse path on or off without rebuilding the model (acceptance sweeps)."""
    self.enable_vsa = bool(enabled)
    for block in self.transformer_blocks:
      attn = block.attn
      if enabled:
        if not isinstance(attn.processor, FastH3AttnProcessor):
          attn.set_processor(FastH3AttnProcessor())
        attn._vsa_meta = attn._vsa_meta if hasattr(attn, "_vsa_meta") else None
      else:
        attn._vsa_meta = None
        if isinstance(attn.processor, FastH3AttnProcessor):
          attn.set_processor(attn._default_processor_cls())


class FastH3ModularPipeline(MiniMaxH3ModularPipeline):
  """MiniMax-H3 modular pipeline that loads the FastH3 transformer and its DMD schedule."""

  contract: Optional[FastH3Contract] = None

  def load_components(self, names=None, workflow=None, **kwargs):
    """Load the components, swapping in the FastH3 transformer and its sigma grid.

    :param names: component names to load, forwarded to the parent.
    :param workflow: workflow whose components to load, forwarded to the parent.
    :param kwargs: extra loading kwargs, forwarded to the parent.
    """
    root = self._pretrained_model_name_or_path
    contract = FastH3Contract.from_checkpoint(root) if root is not None else None
    if contract is not None:
      if contract.task != "t2av":
        raise ValueError(f"FastH3 only ships the {contract.task!r} task; this build supports "
                         "'t2av'.")
      has_gates = _transformer_mounts_vsa_gates(root)
      if has_gates != contract.requires_vsa:
        raise ValueError(f"FastH3 checkpoint at {root} is inconsistent: the contract asks for "
                         f"VSA ({contract.requires_vsa}) but the transformer shards "
                         f"{'carry' if has_gates else 'do not carry'} to_gate_compress weights.")
      spec = self._component_specs.get("transformer")
      if spec is not None:
        spec.type_hint = FastH3Transformer3DModel
      overrides = dict(kwargs)
      enable_vsa = overrides.pop("enable_vsa", None)
      enable_vsa = {} if enable_vsa is None else dict(enable_vsa)
      enable_vsa["transformer"] = has_gates
      overrides["enable_vsa"] = enable_vsa
      overrides.setdefault("vsa_sparsity", {"transformer": contract.vsa_sparsity})
      super().load_components(names=names, workflow=workflow, **overrides)
      self.contract = contract
      transformer = getattr(self, "transformer", None)
      if transformer is not None:
        transformer.refresh_vsa_gate_state()
      self._install_dmd_schedule(contract)
      return None
    self.contract = None
    return super().load_components(names=names, workflow=workflow, **kwargs)

  def _install_dmd_schedule(self, contract: FastH3Contract) -> None:
    """Force the two schedulers onto the student's sigma ladder.

    Idempotency lives in `_wrap_scheduler_timesteps`: it marks every scheduler it wraps. A flag
    on the pipeline would be wrong, because components can be loaded in any order and reloaded
    (`load_components(names=["transformer"])` first, scheduler later, or after
    `unload_components`), and a pipeline-level flag would then skip the ladder silently.
    """
    for name, shift in (("scheduler", contract.video_shift), ("audio_scheduler",
                                                              contract.audio_shift)):
      scheduler = getattr(self, name, None)
      if scheduler is None:
        continue
      sigmas = contract.sigmas(shift)
      _wrap_scheduler_timesteps(scheduler, sigmas, name)


def _wrap_scheduler_timesteps(scheduler, sigmas: torch.Tensor, name: str) -> None:
  """Replace ``scheduler.set_timesteps`` with one that always uses the student's ladder.

  The modular pipeline only forwards `num_inference_steps`, and `MiniMaxH3Scheduler.set_timesteps`
  uses explicit `sigmas` verbatim (no shift, no deduplication), so the contract is injected here.

  :param scheduler: one of the pipeline's two MiniMax-H3 schedulers.
  :param sigmas: shifted sigma grid of that modality.
  :param name: component name, for error messages.
  """
  if getattr(scheduler, "_cache_dit_dmd_sigmas", None) is not None:
    return
  original = scheduler.set_timesteps

  def set_timesteps(num_inference_steps=None, device=None, sigmas_arg=None, **extra):
    if num_inference_steps is not None and int(num_inference_steps) != int(sigmas.numel()):
      raise ValueError(f"FastH3 `{name}` has a {int(sigmas.numel())}-point sigma ladder but the "
                       f"pipeline asked for {num_inference_steps} inference steps.")
    return original(num_inference_steps=None, device=device, sigmas=sigmas.to(device), **extra)

  scheduler.set_timesteps = set_timesteps
  scheduler._cache_dit_dmd_sigmas = sigmas
