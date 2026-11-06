"""The fused MOVA denoising stack that cache-dit caches.

MOVA's released pipeline drives the denoising loop itself: for every layer it conditions the two
towers on each other (the bridge), runs one video block and one audio block, and after
``min(video_layers, audio_layers)`` layers it keeps running video-only blocks. cache-dit can only
skip *whole block calls*, so the loop has to live inside a single block list.

:class:`MOVAStack` therefore takes ownership of the four released components — the two video
experts, the audio tower and the bridge — and rebuilds the loop as **one** ``nn.ModuleList`` of
:class:`MOVAFusedBlock`:

* block ``i < min(video, audio)`` owns ``video_block[i]``, ``video_block_2[i]``, ``audio_block[i]``
  and the two conditioners of layer ``i``;
* block ``i >= min(video, audio)`` owns only the video blocks and passes the audio stream through
  **unchanged** (returning ``None`` would make cache-dit stop updating the second stream's residual
  buffer).

The towers keep their embeddings, rotary tables and output heads; only their ``blocks`` and the
bridge's conditioner dicts are emptied, so each parameter is registered exactly once.

Only the fused stack is exposed to cache-dit (``pipe.transformer``). It is a ``ModelMixin`` because
tensor parallelism asserts ``isinstance(transformer, ModelMixin)`` and regional compilation looks up
``_repeated_blocks`` / ``compile_repeated_blocks``.

Save / load round-trip is **not** supported in this integration: moving the blocks changes the
state-dict layout (``transformer.blocks.N.video_block.*`` instead of ``video_dit.blocks.N.*``), so
quantized checkpoints written from this layout cannot be reloaded by the released loader.
"""

import contextlib
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

from .bridge import DualTowerConditionalBridge, apply_bridge_directions
from .wan_video_dit import WanAudioModel, WanModel, audio_freqs, video_freqs

__all__ = ["MOVAStack", "MOVAFusedBlock"]


def _chunk_sequence(x: torch.Tensor, cp_config, dim: int) -> torch.Tensor:
  """Take this rank's equipartitioned chunk of a full-length tensor.

  The context-parallel hook that splits the hidden states uses the same equipartition, so chunking
  the loop-invariant rotary tables here keeps them aligned with the sharded stream without adding
  them to the hook metadata (a wildcard hook would split them on every block call).

  :param x: Full-length tensor.
  :param cp_config: Armed context-parallel config.
  :param dim: Sequence dimension.
  :returns: The local chunk.
  """

  return x.chunk(cp_config._world_size, dim=dim)[cp_config._rank]


def _fp32_autocast(device: torch.device):
  """Context that forces the timestep path to run in float32, as upstream does.

  :param device: Device the timestep embedding runs on.
  :returns: An autocast context on CUDA, a no-op elsewhere.
  """

  if device.type == "cuda":
    return torch.autocast("cuda", dtype=torch.float32)
  return contextlib.nullcontext()


class MOVAFusedBlock(nn.Module):
  """One layer of the fused MOVA loop.

  The bridge directions run before the two towers' blocks and both read the states that *enter* the
  layer, exactly like the released ``forward_dual_tower_dit``.

  :param index: Zero-based layer index; the cached wrapper keeps the original blocks, so the position
    cannot be recovered by enumerating the (one-element) wrapped list.
  :param video_block: High-noise video block of this layer.
  :param video_block_2: Low-noise video block of this layer, or None.
  :param audio_block: Audio block of this layer, or None for the video-only tail.
  :param a2v_conditioner: Audio->video conditioner of this layer, or None.
  :param v2a_conditioner: Video->audio conditioner of this layer, or None.
  :param condition_scale: Conditioning strength (upstream constant 1.0).
  """

  def __init__(self,
               index: int,
               video_block: nn.Module,
               video_block_2: Optional[nn.Module] = None,
               audio_block: Optional[nn.Module] = None,
               a2v_conditioner: Optional[nn.Module] = None,
               v2a_conditioner: Optional[nn.Module] = None,
               condition_scale: float = 1.0):
    super().__init__()
    self.index = index
    self.video_block = video_block
    self.video_block_2 = video_block_2
    self.audio_block = audio_block
    self.a2v_conditioner = a2v_conditioner
    self.v2a_conditioner = v2a_conditioner
    self.condition_scale = condition_scale

  def forward(self,
              hidden_states: torch.Tensor,
              encoder_hidden_states: torch.Tensor,
              visual_context: torch.Tensor,
              audio_context: torch.Tensor,
              visual_t_mod: torch.Tensor,
              audio_t_mod: torch.Tensor,
              visual_freqs: torch.Tensor,
              audio_freqs_table: torch.Tensor,
              visual_rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
              audio_rope: Optional[Tuple[torch.Tensor, torch.Tensor]],
              use_video_dit_2: bool,
              cp_config=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one fused layer.

    :param hidden_states: Video hidden states, ``[B, S, dim_v]``.
    :param encoder_hidden_states: Audio hidden states, ``[B, L, dim_a]``.
    :param visual_context: Video prompt embedding.
    :param audio_context: Audio prompt embedding.
    :param visual_t_mod: Video timestep modulation, ``[B, 6, dim_v]``.
    :param audio_t_mod: Audio timestep modulation, ``[B, 6, dim_a]``.
    :param visual_freqs: Video self-attention rotary table.
    :param audio_freqs_table: Audio self-attention rotary table.
    :param visual_rope: ``(cos, sin)`` of the video stream for the bridge.
    :param audio_rope: ``(cos, sin)`` of the audio stream for the bridge.
    :param use_video_dit_2: Whether the low-noise expert is active.
    :param cp_config: Armed context-parallel config, or None.
    :returns: ``(video, audio)`` hidden states after the layer.
    """

    video_block = self.video_block_2 if (use_video_dit_2
                                         and self.video_block_2 is not None) else self.video_block

    visual_cond, audio_cond = apply_bridge_directions(
      self.a2v_conditioner,
      self.v2a_conditioner,
      self.condition_scale,
      hidden_states,
      encoder_hidden_states,
      visual_rope=visual_rope,
      audio_rope=audio_rope,
      cp_config=cp_config,
    )

    visual_x = video_block(visual_cond,
                           visual_context,
                           visual_t_mod,
                           visual_freqs,
                           cp_config=cp_config)
    if self.audio_block is None:
      # The tail of the video tower: the audio stream must come back unchanged, never as None.
      return visual_x, encoder_hidden_states
    # The audio stream stays replicated under context parallelism, so its self attention is computed
    # identically on every rank and must not join the collective.
    audio_x = self.audio_block(audio_cond,
                               audio_context,
                               audio_t_mod,
                               audio_freqs_table,
                               cp_config=None)
    return visual_x, audio_x


class MOVAStack(ModelMixin, ConfigMixin):
  """Fused MOVA denoising stack: one forward per CFG branch, both velocities in the return value.

  :param video_dit: High-noise video tower.
  :param video_dit_2: Low-noise video tower, or None for a single-expert run.
  :param audio_dit: Audio tower.
  :param dual_tower_bridge: Cross-modal bridge.
  :param boundary_ratio: Timestep ratio at which the pipeline switches experts; kept for provenance.
  """

  _repeated_blocks = ["MOVAFusedBlock"]
  _supports_gradient_checkpointing = False

  def __init__(self,
               video_dit: WanModel,
               video_dit_2: Optional[WanModel],
               audio_dit: WanAudioModel,
               dual_tower_bridge: DualTowerConditionalBridge,
               boundary_ratio: float = 0.9):
    super().__init__()
    if getattr(dual_tower_bridge, "trainable_condition_scale", False):
      raise NotImplementedError(
        "The fused stack assumes the released constant condition scale; `trainable_condition_scale` "
        "is not supported by this integration.")

    num_visual = len(video_dit.blocks)
    num_audio = len(audio_dit.blocks)
    num_fused = min(num_visual, num_audio)
    condition_scale = float(dual_tower_bridge.condition_scale)

    a2v = dual_tower_bridge.audio_to_video_conditioners
    v2a = dual_tower_bridge.video_to_audio_conditioners
    blocks = []
    for i in range(num_visual):
      blocks.append(
        MOVAFusedBlock(
          index=i,
          video_block=video_dit.blocks[i],
          video_block_2=video_dit_2.blocks[i] if video_dit_2 is not None else None,
          audio_block=audio_dit.blocks[i] if i < num_fused else None,
          a2v_conditioner=a2v[str(i)] if (i < num_fused and str(i) in a2v) else None,
          v2a_conditioner=v2a[str(i)] if (i < num_fused and str(i) in v2a) else None,
          condition_scale=condition_scale,
        ))
    self.blocks = nn.ModuleList(blocks)

    # The blocks now live in the fused stack; emptying the original containers keeps every parameter
    # registered exactly once (a second registration would duplicate the offload handles and the
    # state-dict keys).
    video_dit.blocks = nn.ModuleList()
    if video_dit_2 is not None:
      video_dit_2.blocks = nn.ModuleList()
    audio_dit.blocks = nn.ModuleList()
    dual_tower_bridge.audio_to_video_conditioners = nn.ModuleDict()
    dual_tower_bridge.video_to_audio_conditioners = nn.ModuleDict()

    self.video_dit = video_dit
    self.video_dit_2 = video_dit_2
    self.audio_dit = audio_dit
    self.dual_tower_bridge = dual_tower_bridge

    self.register_to_config(
      num_blocks=num_visual,
      num_visual_layers=num_visual,
      num_audio_layers=num_audio,
      num_fused_layers=num_fused,
      boundary_ratio=boundary_ratio,
    )

  @property
  def num_blocks(self) -> int:
    """Configured block count (``len(self.blocks)`` collapses to 1 under caching)."""

    return self.config.num_blocks

  def visible_layers(self) -> int:
    """Number of layers that carry an audio block.

    :returns: ``min(num_visual_layers, num_audio_layers)``.
    """

    return self.config.num_fused_layers

  def resident_to(self, device) -> "MOVAStack":
    """Move the parts that are not streamed to ``device``.

    With ``layerwise_offload_filter`` only the fused blocks are offload targets, so cache-dit leaves
    every other leaf on the host. The embeddings, the heads and the bridge's rotary module run on
    every step (and their activations feed the context-parallel gather), so they must stay resident.

    :param device: Compute device.
    :returns: ``self``, for chaining.
    """

    for tower in (self.video_dit, self.video_dit_2, self.audio_dit, self.dual_tower_bridge):
      if tower is not None:
        tower.to(device)
    return self

  @staticmethod
  def layerwise_offload_filter(name: str, module: torch.nn.Module) -> bool:
    """Select one fused block as the streaming unit.

    :param name: Module name inside the transformer.
    :param module: Candidate submodule.
    :returns: Whether ``module`` should be streamed.
    """

    del name
    return isinstance(module, MOVAFusedBlock)

  def forward(self,
              visual_latents: torch.Tensor,
              audio_latents: torch.Tensor,
              context: torch.Tensor,
              timestep: torch.Tensor,
              audio_context: Optional[torch.Tensor] = None,
              audio_timestep: Optional[torch.Tensor] = None,
              video_fps: float = 24.0,
              use_video_dit_2: bool = False,
              cp_config=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one denoising step and return both velocities.

    Everything in here runs on every step: only the block loop is inside the cached region.

    :param visual_latents: ``[B, C, T, H, W]`` video latents concatenated with the conditioning
      channels (the pipeline builds the concatenation).
    :param audio_latents: ``[B, latent_dim, T_a]`` audio latents.
    :param context: Prompt embedding, ``[B, L, text_dim]``.
    :param timestep: ``[B]`` video timestep.
    :param audio_context: Prompt embedding for the audio tower; defaults to ``context``.
    :param audio_timestep: ``[B]`` audio timestep; defaults to ``timestep``.
    :param video_fps: Request frame rate, used to align the cross-modal rotary tables.
    :param use_video_dit_2: Whether the low-noise expert is active for this step.
    :param cp_config: Armed context-parallel config, or None.
    :returns: ``(video velocity, audio velocity)`` in latent layout.
    """

    # The CP runtime arms the transformer instance (`_cp_config`); the forward argument is only used
    # by callers that drive the stack directly.
    if cp_config is None:
      cp_config = getattr(self, "_cp_config", None)
    if cp_config is not None and getattr(cp_config, "_world_size", 1) <= 1:
      cp_config = None

    visual_dit = self.video_dit_2 if (use_video_dit_2
                                      and self.video_dit_2 is not None) else self.video_dit
    if audio_context is None:
      audio_context = context
    if audio_timestep is None:
      audio_timestep = timestep

    with _fp32_autocast(visual_latents.device):
      visual_t, visual_t_mod = visual_dit.time_embed(timestep)
      audio_t, audio_t_mod = self.audio_dit.time_embed(audio_timestep)

    dtype = visual_dit.dtype
    visual_t = visual_t.to(dtype)
    visual_t_mod = visual_t_mod.to(dtype)
    audio_t = audio_t.to(dtype)
    audio_t_mod = audio_t_mod.to(dtype)

    visual_context = visual_dit.text_embedding(context)
    audio_context = self.audio_dit.text_embedding(audio_context)

    visual_x, grid_size = visual_dit.patchify(visual_latents.to(dtype))
    audio_x, audio_len = self.audio_dit.patchify(audio_latents.to(dtype))

    visual_freqs_table = video_freqs(visual_dit.freqs, grid_size[0], grid_size[1], grid_size[2],
                                     visual_x.device)
    audio_freqs_table = audio_freqs(self.audio_dit.freqs, audio_len[0], audio_x.device)

    visual_rope = None
    audio_rope = None
    if self.dual_tower_bridge.apply_cross_rope:
      visual_rope, audio_rope = self.dual_tower_bridge.build_aligned_freqs(
        video_fps=video_fps,
        grid_size=grid_size,
        audio_steps=audio_x.shape[1],
        device=visual_x.device,
        dtype=visual_x.dtype,
      )

    if cp_config is not None and cp_config._world_size > 1:
      world_size = cp_config._world_size
      if visual_x.shape[1] % world_size != 0:
        raise ValueError(
          f"Context parallelism needs a video sequence divisible by the CP degree: got "
          f"{visual_x.shape[1]} video tokens for cp_size={world_size}. Lower the CP degree or change "
          f"the geometry; the audio stream ({audio_x.shape[1]} tokens) stays replicated and is not "
          f"split.")
      # Only the video stream is sharded; its rotary tables follow the shard, the audio ones (and the
      # bridge's audio rope) stay full because the audio stream is replicated.
      visual_freqs_table = _chunk_sequence(visual_freqs_table, cp_config, dim=0)
      if visual_rope is not None:
        visual_rope = tuple(_chunk_sequence(t, cp_config, dim=1) for t in visual_rope)

    visual_x, audio_x = self._run_blocks(
      visual_x,
      audio_x,
      visual_context,
      audio_context,
      visual_t_mod,
      audio_t_mod,
      visual_freqs_table,
      audio_freqs_table,
      visual_rope,
      audio_rope,
      use_video_dit_2,
      cp_config,
    )

    visual_out = visual_dit.unpatchify(visual_dit.head(visual_x, visual_t), grid_size)
    audio_out = self.audio_dit.unpatchify(self.audio_dit.head(audio_x, audio_t), audio_len)
    return visual_out, audio_out

  def _run_blocks(self, visual_x, audio_x, visual_context, audio_context, visual_t_mod, audio_t_mod,
                  visual_freqs_table, audio_freqs_table, visual_rope, audio_rope, use_video_dit_2,
                  cp_config):
    """Walk the fused block list once.

    Kept separate so the cached wrapper (which swaps ``self.blocks`` for a one-element list holding
    the cache wrapper) only has to intercept this loop.
    """

    for block in self.blocks:
      visual_x, audio_x = block(
        visual_x,
        audio_x,
        visual_context,
        audio_context,
        visual_t_mod,
        audio_t_mod,
        visual_freqs_table,
        audio_freqs_table,
        visual_rope,
        audio_rope,
        use_video_dit_2,
        cp_config,
      )
    return visual_x, audio_x
