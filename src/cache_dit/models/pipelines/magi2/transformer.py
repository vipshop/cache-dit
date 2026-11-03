"""Native MAGI-2 Preview diffusion transformer (114B total, 6B active per token).

Adapted from ``vllm-omni @ 7f217a1`` ``vllm_omni/diffusion/models/magi2/modeling_magi2.py``
(Apache-2.0, Copyright (c) 2026 SandAI). Intentional differences:

* the module tree, class names, attribute names and state-dict keys are unchanged, so the released
  ``preview/model.safetensors`` shards load without renaming;
* the packed attention kernel is called directly (:mod:`.attention`) instead of going through
  ``vllm_omni``'s ``Attention`` layer and its metadata side channel;
* ``load_weights`` is a strict loader over a safetensors shard index, driven by
  :meth:`Magi2Pipeline.from_pretrained`, instead of vLLM's ``ComponentSource`` plumbing;
* implemented for cache-dit: ``_repeated_blocks`` declares the block stack (regional quantization and
  ``compile_repeated_blocks`` discover it), ``_layerwise_offload_blocks_attrs`` names the streamable
  container, and ``forward`` accepts an already-sharded packed sequence;
* cache-dit style (2-space indent, Sphinx docstrings).
"""

import os

import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin

from .attention import Magi2Attention, VarlenHandler
from .common_layers import ElementWiseFourierEmbed, ModalityDispatcher, make_grouped_linear
from .configuration import Magi2PreviewConfig
from .layers import MHCHandler, Magi2MultiHeadMoE, Magi2MultiHeadMoEConfig, MultiModalityRMSNorm, swiglu7
from .parallel import Magi2SequenceDispatcher

__all__ = [
  "Modality",
  "Magi2MLP",
  "Magi2MultiHeadMoELayer",
  "Magi2PostAdapter",
  "Magi2PreAdapter",
  "Magi2PreviewTransformer",
  "Magi2TransformerBlock",
  "Magi2TransformerLayer",
]


class Modality:
  """Modality ids of the packed token sequence."""

  VIDEO = 0
  AUDIO = 1
  TEXT = 2
  TIME = 3


class Magi2MLP(nn.Module):
  """Dense SwiGLU7 MLP.

  :param config: Preview architecture config.
  :param num_modality: Number of per-modality weight groups.
  """

  def __init__(self, config: Magi2PreviewConfig, *, num_modality: int) -> None:
    super().__init__()
    intermediate_size = int(config.hidden_size * config.intermediate_factor * 2 / 3) // 128 * 128
    # Tiny unit-test configurations can be narrower than the release's 128-channel rounding unit
    # while keeping the same formula family.
    if intermediate_size == 0:
      intermediate_size = max(1, int(config.hidden_size * config.intermediate_factor * 2 / 3))
    self.intermediate_size = intermediate_size
    self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)
    self.up_gate_proj = make_grouped_linear(
      config.hidden_size,
      2 * intermediate_size,
      num_experts=num_modality,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="column",
    )
    self.down_proj = make_grouped_linear(
      intermediate_size,
      config.hidden_size,
      num_experts=num_modality,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="row",
    )

  def forward(self, hidden_states: torch.Tensor, dispatcher: ModalityDispatcher) -> torch.Tensor:
    """Run the dense MLP.

    :param hidden_states: ``[tokens, hidden]`` input.
    :param dispatcher: Modality grouping of the packed tokens.
    :returns: ``[tokens, hidden]`` output.
    """

    hidden_states = self.pre_norm(hidden_states, dispatcher)
    hidden_states = self.up_gate_proj(hidden_states, dispatcher)
    hidden_states = swiglu7(hidden_states)
    return self.down_proj(hidden_states, dispatcher)


class Magi2MultiHeadMoELayer(nn.Module):
  """Multi-head MoE MLP: routed experts plus a shared and a per-modality shared expert.

  :param config: Preview architecture config.
  """

  def __init__(self, config: Magi2PreviewConfig) -> None:
    super().__init__()
    moe = config.moe
    self.config = config
    self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=3)
    self.split_linear = make_grouped_linear(
      config.hidden_size,
      config.hidden_size,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="column",
    )
    self.moe_mlp = Magi2MultiHeadMoE(
      Magi2MultiHeadMoEConfig(
        hidden_size=config.hidden_size,
        num_heads=moe.num_heads,
        num_experts=moe.num_experts,
        top_k=moe.top_k,
        expert_intermediate_size=moe.expert_intermediate_size,
        params_dtype=config.params_dtype,
        score_func=moe.score_function,
        route_norm=moe.normalize_routing_weights,
        route_scale=moe.routing_scale,
      ))
    self.merge_linear = make_grouped_linear(
      config.hidden_size,
      config.hidden_size,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="row",
    )
    self.shared_expert_fc1 = make_grouped_linear(
      config.hidden_size,
      2 * moe.shared_expert_intermediate_size,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="column",
    )
    self.shared_expert_fc2 = make_grouped_linear(
      moe.shared_expert_intermediate_size,
      config.hidden_size,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="row",
    )
    self.modality_specific_shared_expert_fc1 = make_grouped_linear(
      config.hidden_size,
      2 * moe.modality_shared_expert_intermediate_size,
      num_experts=3,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="column",
    )
    self.modality_specific_shared_expert_fc2 = make_grouped_linear(
      moe.modality_shared_expert_intermediate_size,
      config.hidden_size,
      num_experts=3,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="row",
    )

  def _shared_experts(self, normalized: torch.Tensor,
                      dispatcher: ModalityDispatcher) -> torch.Tensor:
    """Run the always-on shared experts.

    :param normalized: Normalized MoE input.
    :param dispatcher: Modality grouping of the packed tokens.
    :returns: Sum of both shared-expert outputs.
    """

    # SwiGLU7 acts on adjacent column pairs, so activating each projection separately keeps both
    # GEMM outputs and the down-projection inputs dense.
    shared = swiglu7(self.shared_expert_fc1(normalized))
    modality = swiglu7(self.modality_specific_shared_expert_fc1(normalized, dispatcher))
    return self.shared_expert_fc2(shared) + self.modality_specific_shared_expert_fc2(
      modality, dispatcher)

  def route_input(self, hidden_states: torch.Tensor, dispatcher: ModalityDispatcher):
    """Normalize and split the routed-expert input.

    :param hidden_states: Block input.
    :param dispatcher: Modality grouping of the packed tokens.
    :returns: ``(normalized, routed)``.
    """

    normalized = self.pre_norm(hidden_states, dispatcher)
    return normalized, self.split_linear(normalized)

  def combine(self, normalized: torch.Tensor, routed: torch.Tensor, dispatcher: ModalityDispatcher):
    """Merge the routed experts with the shared experts.

    :param normalized: Normalized MoE input.
    :param routed: Routed-expert output.
    :param dispatcher: Modality grouping of the packed tokens.
    :returns: ``[tokens, hidden]`` MoE output.
    """

    return self.merge_linear(routed) + self._shared_experts(normalized, dispatcher)


class Magi2PreAdapter(nn.Module):
  """Per-modality input projections and the element-wise Fourier rotary embed.

  :param config: Preview architecture config.
  """

  def __init__(self, config: Magi2PreviewConfig) -> None:
    super().__init__()
    self.config = config
    self.adapter_dim = config.virtual_width
    self.video_embedder = nn.Linear(config.video_in_channels,
                                    self.adapter_dim,
                                    bias=True,
                                    dtype=torch.float32)
    self.text_embedder = nn.Linear(config.text_in_channels,
                                   self.adapter_dim,
                                   bias=True,
                                   dtype=torch.float32)
    self.audio_embedder = nn.Linear(config.audio_in_channels,
                                    self.adapter_dim,
                                    bias=True,
                                    dtype=torch.float32)
    self.rope = ElementWiseFourierEmbed(config.head_dim, learnable=False)

  def forward(
    self,
    packed: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
    text_indices: torch.Tensor,
  ) -> torch.Tensor:
    """Project every packed token into the mHC stream stack.

    :param packed: ``[tokens, max_channels]`` packed input.
    :param video_indices: Token positions of the video modality.
    :param audio_indices: Token positions of the audio modality.
    :param text_indices: Token positions of the text modality.
    :returns: ``[tokens, virtual_width]`` projected tokens.
    """

    # Every token belongs to exactly one modality, so the three copies below cover the whole buffer.
    # Embedding in FP32 and rounding each modality once matches the released cast at the block
    # boundary.
    output = torch.empty(
      packed.shape[0],
      self.adapter_dim,
      dtype=self.config.params_dtype,
      device=packed.device,
    )
    for indices, embedder, in_channels in (
      (text_indices, self.text_embedder, self.config.text_in_channels),
      (audio_indices, self.audio_embedder, self.config.audio_in_channels),
      (video_indices, self.video_embedder, self.config.video_in_channels),
    ):
      if indices.numel():
        tokens = packed[:, :in_channels].index_select(0, indices).float()
        output.index_copy_(0, indices, embedder(tokens).to(output.dtype))
    return output


class Magi2PostAdapter(nn.Module):
  """Per-modality output projections.

  :param config: Preview architecture config.
  """

  def __init__(self, config: Magi2PreviewConfig) -> None:
    super().__init__()
    self.config = config
    self.adapter_dim = config.virtual_width
    self.final_norm_video = MultiModalityRMSNorm(self.adapter_dim)
    self.final_norm_audio = MultiModalityRMSNorm(self.adapter_dim)
    self.final_linear_video = nn.Linear(self.adapter_dim,
                                        config.video_in_channels,
                                        bias=False,
                                        dtype=torch.float32)
    self.final_linear_audio = nn.Linear(self.adapter_dim,
                                        config.audio_in_channels,
                                        bias=False,
                                        dtype=torch.float32)
    self.final_out_dim = max(config.video_in_channels, config.audio_in_channels)

  def forward(
    self,
    hidden_states: torch.Tensor,
    video_indices: torch.Tensor,
    audio_indices: torch.Tensor,
  ) -> torch.Tensor:
    """Project the mHC stack back to video and audio channel chunks.

    :param hidden_states: ``[tokens, virtual_width]`` stack output.
    :param video_indices: Token positions of the video modality.
    :param audio_indices: Token positions of the audio modality.
    :returns: ``[tokens, max(video_channels, audio_channels)]`` FP32 output.
    """

    output = torch.zeros(
      hidden_states.shape[0],
      self.final_out_dim,
      dtype=torch.float32,
      device=hidden_states.device,
    )
    if video_indices.numel():
      video = self.final_norm_video(hidden_states.index_select(0, video_indices).float())
      video = self.final_linear_video(video).float()
      output[:, :self.config.video_in_channels].index_copy_(0, video_indices, video)
    if audio_indices.numel():
      audio = self.final_norm_audio(hidden_states.index_select(0, audio_indices).float())
      audio = self.final_linear_audio(audio).float()
      output[:, :self.config.audio_in_channels].index_copy_(0, audio_indices, audio)
    return output


class Magi2TransformerLayer(nn.Module):
  """One MAGI-2 layer: attention, mHC hyper-connect and a dense or MoE MLP.

  The layer is the unit cache-dit caches: its signature is
  ``forward(hidden_states, rope, varlen_handler, modality_dispatcher, cp_split_sizes)`` and it
  returns one tensor (``ForwardPattern.Pattern_3``).

  :param config: Preview architecture config.
  :param layer_index: Zero-based depth index; selects the MoE/dense and multimodal variants.
  """

  def __init__(self, config: Magi2PreviewConfig, layer_index: int) -> None:
    super().__init__()
    num_modality = 3 if layer_index in config.multimodal_layers else 1
    self.config = config
    self.layer_index = layer_index
    self.attention = Magi2Attention(config, num_modality=num_modality)
    if layer_index in config.moe.layers:
      self.mlp = Magi2MultiHeadMoELayer(config)
      self.region_methods = ("_attention_input", "_moe_input", "_moe_output")
    else:
      self.mlp = Magi2MLP(config, num_modality=num_modality)
      self.region_methods = ("_attention_input", "_dense_output")
    self._init_mhc(num_modality)

  def _init_mhc(self, num_modality: int) -> None:
    """Register the mHC parameters of both branches.

    :param num_modality: Number of per-modality weight groups.
    """

    streams = self.config.mhc.num_streams
    hidden = self.config.hidden_size
    alpha = self.config.mhc.alpha_init
    for branch in ("attn", "mlp"):
      for coefficient in ("pre", "post", "res"):
        self.register_parameter(
          f"mhc_alpha_{coefficient}_{branch}",
          nn.Parameter(torch.full((1, ), alpha, dtype=torch.float32)),
        )
      self.register_parameter(
        f"mhc_bias_pre_{branch}",
        nn.Parameter(torch.empty(streams, dtype=torch.float32)),
      )
      self.register_parameter(
        f"mhc_bias_post_{branch}",
        nn.Parameter(torch.empty(streams, dtype=torch.float32)),
      )
      self.register_parameter(
        f"mhc_bias_res_{branch}",
        nn.Parameter(torch.empty(streams, streams, dtype=torch.float32)),
      )
      self.register_parameter(
        f"mhc_phi_fused_{branch}",
        nn.Parameter(
          torch.empty(
            streams * hidden,
            streams + streams + streams * streams,
            dtype=torch.float32,
          )),
      )
    self.mhc_norm = MultiModalityRMSNorm(
      streams * hidden,
      num_modality=num_modality,
      out_dtype=torch.float32,
    )
    self.mhc_handler = MHCHandler(
      streams,
      hidden,
      sinkhorn_iterations=self.config.mhc.sinkhorn_iterations,
      sinkhorn_epsilon=self.config.mhc.sinkhorn_epsilon,
    )

  def _branch_logits(self, streams: torch.Tensor, branch: str, dispatcher: ModalityDispatcher):
    return self.mhc_handler.compute_logits(
      self.mhc_handler.flatten(streams),
      lambda flattened: self.mhc_norm(flattened, dispatcher),
      getattr(self, f"mhc_phi_fused_{branch}"),
    )

  def _branch_input(self, streams: torch.Tensor, branch: str, logits) -> torch.Tensor:
    pre_logits, _, _ = logits
    return self.mhc_handler.apply_pre(
      streams,
      (
        getattr(self, f"mhc_alpha_pre_{branch}"),
        getattr(self, f"mhc_bias_pre_{branch}"),
        pre_logits,
      ),
      out_dtype=streams.dtype,
    )

  def _connect(self, streams: torch.Tensor, output: torch.Tensor, branch: str,
               logits) -> torch.Tensor:
    _, post_logits, residual_logits = logits
    post, residual = self.mhc_handler.compute_post_residual(
      (
        getattr(self, f"mhc_alpha_post_{branch}"),
        getattr(self, f"mhc_bias_post_{branch}"),
        post_logits,
      ),
      (
        getattr(self, f"mhc_alpha_res_{branch}"),
        getattr(self, f"mhc_bias_res_{branch}"),
        residual_logits,
      ),
      out_dtype=streams.dtype,
    )
    return self.mhc_handler.hyper_connect(streams, output, post, residual)

  def _attention_input(self, hidden_states: torch.Tensor, rope: torch.Tensor, modality_dispatcher):
    """First compile region: mHC pre-mix and Q/K/V projection.

    :param hidden_states: ``[tokens, virtual_width]`` layer input.
    :param rope: ``[tokens, head_dim]`` rotary table.
    :param modality_dispatcher: Modality grouping of the packed tokens.
    :returns: ``(streams, attention_logits, q, k, v, gates)``.
    """

    streams = hidden_states.reshape(hidden_states.shape[0], self.config.mhc.num_streams,
                                    self.config.hidden_size)
    attention_logits = self._branch_logits(streams, "attn", modality_dispatcher)
    attention_input = self._branch_input(streams, "attn", attention_logits)
    q, k, v, gates = self.attention.project(attention_input, rope, modality_dispatcher)
    return streams, attention_logits, q, k, v, gates

  def _mlp_input(self, streams: torch.Tensor, attention: torch.Tensor, gates: torch.Tensor,
                 attention_logits, modality_dispatcher):
    """Apply the attention branch and prepare the MLP branch.

    :param streams: ``[tokens, streams, hidden]`` residual stack.
    :param attention: Attention output.
    :param gates: Per-head attention gate logits.
    :param attention_logits: Logits of the attention branch.
    :param modality_dispatcher: Modality grouping of the packed tokens.
    :returns: ``(streams, mlp_logits, mlp_input)``.
    """

    attention_output = self.attention.output(attention, gates, modality_dispatcher)
    streams = self._connect(streams, attention_output, "attn", attention_logits)
    mlp_logits = self._branch_logits(streams, "mlp", modality_dispatcher)
    return streams, mlp_logits, self._branch_input(streams, "mlp", mlp_logits)

  def _dense_output(self, streams: torch.Tensor, attention: torch.Tensor, gates: torch.Tensor,
                    attention_logits, modality_dispatcher) -> torch.Tensor:
    """Second compile region of a dense layer.

    :param streams: ``[tokens, streams, hidden]`` residual stack.
    :param attention: Attention output.
    :param gates: Per-head attention gate logits.
    :param attention_logits: Logits of the attention branch.
    :param modality_dispatcher: Modality grouping of the packed tokens.
    :returns: ``[tokens, virtual_width]`` layer output.
    """

    streams, mlp_logits, mlp_input = self._mlp_input(streams, attention, gates, attention_logits,
                                                     modality_dispatcher)
    streams = self._connect(streams, self.mlp(mlp_input, modality_dispatcher), "mlp", mlp_logits)
    return streams.reshape(streams.shape[0], -1)

  def _moe_input(self, streams: torch.Tensor, attention: torch.Tensor, gates: torch.Tensor,
                 attention_logits, modality_dispatcher):
    """Second compile region of an MoE layer up to the routed-expert input.

    :param streams: ``[tokens, streams, hidden]`` residual stack.
    :param attention: Attention output.
    :param gates: Per-head attention gate logits.
    :param attention_logits: Logits of the attention branch.
    :param modality_dispatcher: Modality grouping of the packed tokens.
    :returns: ``(streams, mlp_logits, normalized, routed)``.
    """

    streams, mlp_logits, mlp_input = self._mlp_input(streams, attention, gates, attention_logits,
                                                     modality_dispatcher)
    normalized, routed = self.mlp.route_input(mlp_input, modality_dispatcher)
    return streams, mlp_logits, normalized, routed

  def _moe_output(self, streams: torch.Tensor, mlp_logits, normalized: torch.Tensor,
                  routed: torch.Tensor, modality_dispatcher) -> torch.Tensor:
    """Second compile region of an MoE layer after the experts.

    :param streams: ``[tokens, streams, hidden]`` residual stack.
    :param mlp_logits: Logits of the MLP branch.
    :param normalized: Normalized MoE input.
    :param routed: Routed-expert output.
    :param modality_dispatcher: Modality grouping of the packed tokens.
    :returns: ``[tokens, virtual_width]`` layer output.
    """

    mlp_output = self.mlp.combine(normalized, routed, modality_dispatcher)
    streams = self._connect(streams, mlp_output, "mlp", mlp_logits)
    return streams.reshape(streams.shape[0], -1)

  def forward(
    self,
    hidden_states: torch.Tensor,
    rope: torch.Tensor,
    varlen_handler: VarlenHandler,
    modality_dispatcher: ModalityDispatcher,
    cp_split_sizes,
  ) -> torch.Tensor:
    """Run one layer.

    :param hidden_states: ``[tokens, virtual_width]`` layer input.
    :param rope: ``[tokens, head_dim]`` rotary table.
    :param varlen_handler: Packed-sequence metadata.
    :param modality_dispatcher: Modality grouping of the packed tokens.
    :param cp_split_sizes: Per-rank token counts of the full sequence.
    :returns: ``[tokens, virtual_width]`` layer output.
    """

    streams, attention_logits, q, k, v, gates = self._attention_input(hidden_states, rope,
                                                                      modality_dispatcher)
    attention = self.attention.attend(q, k, v, varlen_handler, cp_split_sizes)
    if not isinstance(self.mlp, Magi2MultiHeadMoELayer):
      return self._dense_output(streams, attention, gates, attention_logits, modality_dispatcher)
    streams, mlp_logits, normalized, routed = self._moe_input(streams, attention, gates,
                                                              attention_logits, modality_dispatcher)
    routed = self.mlp.moe_mlp(routed)
    return self._moe_output(streams, mlp_logits, normalized, routed, modality_dispatcher)


class Magi2TransformerBlock(nn.Module):
  """The repeated layer stack; this is the module cache-dit caches.

  :param config: Preview architecture config.
  """

  def __init__(self, config: Magi2PreviewConfig) -> None:
    super().__init__()
    self.config = config
    self.layers = nn.ModuleList(
      Magi2TransformerLayer(config, index) for index in range(config.num_layers))

  def __iter__(self):
    """Expose the nested layers to layerwise offload backends."""

    return iter(self.layers)

  def shard_for_tp(self, tp_group) -> None:
    """Shard every layer's grouped projections and MoE head blocks in place.

    cache-dit parallelizes the module the block adapter points at, which is this container, so the
    planner calls the sharding here rather than on the outer transformer.

    :param tp_group: Parameter-parallel group installed by the TP planner.
    """

    if tp_group.world_size <= 1:
      return
    for layer in self.layers:
      layer.attention.shard_for_tp(tp_group)
      for module in layer.mlp.modules():
        if isinstance(module, Magi2MultiHeadMoE):
          module.shard_for_tp(tp_group)
        elif hasattr(module, "shard_for_tp"):
          module.shard_for_tp(tp_group)

  def forward(
    self,
    hidden_states: torch.Tensor,
    rope: torch.Tensor,
    varlen_handler: VarlenHandler,
    modality_dispatcher: ModalityDispatcher,
    cp_split_sizes,
  ) -> torch.Tensor:
    """Run the layer stack on modality-major tokens.

    :param hidden_states: ``[tokens, virtual_width]`` stack input.
    :param rope: ``[tokens, head_dim]`` rotary table.
    :param varlen_handler: Packed-sequence metadata.
    :param modality_dispatcher: Modality grouping of the packed tokens.
    :param cp_split_sizes: Per-rank token counts of the full sequence.
    :returns: ``[tokens, virtual_width]`` stack output in packed order.
    """

    hidden_states = modality_dispatcher.permute(hidden_states)
    for layer in self.layers:
      hidden_states = layer(
        hidden_states,
        rope,
        varlen_handler,
        modality_dispatcher,
        cp_split_sizes,
      )
    return modality_dispatcher.inverse_permute(hidden_states)


class Magi2PreviewTransformer(ModelMixin):
  """Native preview DiT with the released checkpoint hierarchy.

  The module tree is ``pre_adapter`` / ``block.layers.*`` / ``post_adapter``, matching the released
  ``preview/model.safetensors`` keys one to one. ``config`` is the frozen
  :class:`~.configuration.Magi2PreviewConfig` dataclass rather than a diffusers ``FrozenDict``: the
  released checkpoint has no diffusers ``config.json``, and ``num_layers`` has to stay readable while
  the cached stack holds a single wrapped block.

  :param config: Preview architecture config.
  """

  _repeated_blocks = ["Magi2TransformerLayer"]
  # Upstream's dynamic-offload backend reads this name; cache-dit does not. Keep it for provenance and
  # use `layerwise_offload_filter` with cache-dit's own API (its default leaf-module discovery would
  # leave the multi-head MoE parameter blocks unassigned and try to pin 213 GiB on one device).
  _layerwise_offload_blocks_attrs = ["block"]
  _supports_gradient_checkpointing = False

  def resident_to(self, device) -> "Magi2PreviewTransformer":
    """Move the non-streamed parts (both adapters and the rotary bands) to ``device``.

    With `layerwise_offload_filter` only the layers are offload targets, and cache-dit leaves a leaf
    module that no target owns on the host. The adapters run on every step, so they have to be
    resident; the activations they produce then live on the compute device, which is also what the
    CP gather needs.

    :param device: Compute device.
    :returns: ``self``, for chaining.
    """

    self.pre_adapter.to(device)
    self.post_adapter.to(device)
    return self

  @staticmethod
  def layerwise_offload_filter(name: str, module: torch.nn.Module) -> bool:
    """Select the streaming unit for cache-dit's layerwise offload.

    One target per transformer layer (~6 GiB, the MoE layers), which is also the cache-dit block
    granularity: a skipped cached layer then costs no host-to-device transfer at all.

    :param name: Module name inside the transformer.
    :param module: Candidate submodule.
    :returns: Whether ``module`` should be streamed.
    """

    return isinstance(module, Magi2TransformerLayer)

  def __init__(self, config: Magi2PreviewConfig = None) -> None:
    super().__init__()
    config = config or Magi2PreviewConfig()
    config.validate()
    self.config = config
    self.pre_adapter = Magi2PreAdapter(config)
    self.post_adapter = Magi2PostAdapter(config)
    self.block = Magi2TransformerBlock(config)
    # Set once the CP runtime installs its device mesh; the packed sequence is sharded by the model
    # itself, so the group has to reach the kernels before the first dispatch.
    self._magi2_group_tag = None

  @property
  def layers(self) -> nn.ModuleList:
    """Expose the repeated layers for direct access."""

    return self.block.layers

  def dispatch_scale(self) -> int:
    """Return the token-sharding factor of the installed Ulysses group."""

    from .parallel import get_magi2_ulysses_group

    return get_magi2_ulysses_group().world_size

  def _ensure_parallel_groups(self) -> None:
    """Install the Ulysses group that cache-dit passed in as ``_cp_config``.

    The CP planner returns an empty hook plan on purpose: MAGI-2 shards the packed sequence inside
    its own ``forward``, so cache-dit only has to hand over the process group.
    """

    cp_config = getattr(self, "_cp_config", None)
    if cp_config is None:
      # cache-dit parallelizes the module the block adapter points at, i.e. the inner block.
      cp_config = getattr(self.block, "_cp_config", None)
    if cp_config is None or getattr(cp_config, "_world_size", 1) <= 1:
      return
    mesh = getattr(cp_config, "_ulysses_mesh", None)
    if mesh is None:
      return
    tag = (cp_config._world_size, cp_config._rank)
    if self._magi2_group_tag == tag:
      return
    from .parallel import set_magi2_groups

    set_magi2_groups(ulysses_group=mesh.get_group())
    self._magi2_group_tag = tag

  def forward(
    self,
    x: torch.Tensor,
    coords_mapping: torch.Tensor,
    modality_mapping: torch.Tensor,
    varlen_handler: VarlenHandler,
    time_token_sequence: torch.Tensor = None,
  ) -> torch.Tensor:
    """Run one denoising step.

    The packed sequence is sharded by the model itself (Ulysses context parallelism): the split
    happens before the cached layer stack and the gather after it, so the cache wrapper never sees a
    partially gathered sequence.

    :param x: ``[tokens, max_channels]`` packed noisy tokens.
    :param coords_mapping: ``[tokens, 9]`` coordinates.
    :param modality_mapping: ``[tokens]`` modality ids.
    :param varlen_handler: Packed-sequence metadata over the full sequence.
    :param time_token_sequence: Optional ``[tokens, time_channel_dim]`` time channel.
    :returns: ``[tokens, max(video_channels, audio_channels)]`` predicted velocity.
    """

    self._ensure_parallel_groups()
    dispatcher = Magi2SequenceDispatcher()
    x = dispatcher.dispatch(x)
    coords_mapping = dispatcher.dispatch(coords_mapping)
    modality_mapping = dispatcher.dispatch(modality_mapping)
    if time_token_sequence is not None:
      time_token_sequence = dispatcher.dispatch(time_token_sequence)
    assert dispatcher.split_sizes is not None
    cp_split_sizes = dispatcher.split_sizes

    rope = self.pre_adapter.rope(coords_mapping)
    time_mask = modality_mapping == int(Modality.TIME)
    modality_mapping = torch.where(time_mask, int(Modality.TEXT), modality_mapping)
    modality_dispatcher = ModalityDispatcher(modality_mapping, 3)
    video_indices = torch.nonzero(modality_mapping == int(Modality.VIDEO)).flatten()
    audio_indices = torch.nonzero(modality_mapping == int(Modality.AUDIO)).flatten()
    text_indices = torch.nonzero(modality_mapping == int(Modality.TEXT)).flatten()

    hidden_states = self.pre_adapter(x, video_indices, audio_indices, text_indices)
    if time_token_sequence is not None and time_token_sequence.shape[-1] > 0:
      hidden_states[:, :time_token_sequence.shape[-1]] = time_token_sequence.to(hidden_states.dtype)
    hidden_states = self.block(
      hidden_states,
      rope,
      varlen_handler,
      modality_dispatcher,
      cp_split_sizes,
    )
    # A layerwise-offload post-hook returns the activations to the device the weights were parked on
    # (the host), so bring the stack output back to the resident compute device before the output
    # head and the sequence gather (a collective must run on the process group's backend device).
    compute_device = self.pre_adapter.video_embedder.weight.device
    if hidden_states.device != compute_device:
      hidden_states = hidden_states.to(compute_device)
    output = self.post_adapter(hidden_states, video_indices, audio_indices)
    return dispatcher.undispatch(output)

  def shard_for_tp(self, tp_group) -> None:
    """Shard every replicated projection and MoE head block for tensor parallelism.

    MAGI-2 shards the grouped-linear output/input axis (column/row) and the MoE *head* axis, and the
    released checkpoint already stores the flattened ``(head, expert)`` layout, so no weight
    rearrangement is needed - only an in-place slice of the CPU-resident weights.

    :param tp_group: Parameter-parallel group installed by the TP planner.
    """

    if tp_group.world_size <= 1:
      return
    if getattr(self, "_is_tp_sharded", False):
      raise RuntimeError("MAGI-2 transformer is already sharded for tensor parallelism")

    from .parallel import set_magi2_groups

    set_magi2_groups(tp_group=tp_group)
    self.block.shard_for_tp(tp_group)
    self._is_tp_sharded = True

  def _moe_for_weight(self, name: str):
    """Return the MoE module owning ``name``, if any.

    :param name: State-dict key.
    :returns: The MoE module, or ``None``.
    """

    if ".moe_mlp." not in name:
      return None
    module_name = name.split(".moe_mlp.", 1)[0] + ".moe_mlp"
    module = dict(self.named_modules()).get(module_name)
    return module if isinstance(module, Magi2MultiHeadMoE) else None

  _EP_SHARDED_SUFFIXES = (
    ".moe_mlp.gate",
    ".moe_mlp.W_gate",
    ".moe_mlp.W_up",
    ".moe_mlp.W_down",
    ".moe_mlp.router.expert_bias",
    ".moe_mlp.router.expert_bias_ema",
  )

  def load_weights(self, weights) -> set:
    """Strictly load and TP/MoE-slice one shard of the released checkpoint.

    A 213 GiB checkpoint is streamed shard by shard, so completeness is verified by
    :meth:`finalize_loading` after the last shard instead of after every call.

    :param weights: Iterable of ``(name, tensor)`` checkpoint pairs, prefix already stripped.
    :returns: The set of loaded parameter names.
    :raises ValueError: On an unexpected name or a shape mismatch.
    """

    targets = self._weight_targets()
    loaded = set()
    for raw_name, checkpoint_tensor in weights:
      name = raw_name.removeprefix("transformer.")
      if name not in targets:
        raise ValueError(f"unexpected MAGI-2 preview checkpoint weight {raw_name!r}")
      target = targets[name]
      checkpoint_transform = getattr(target, "checkpoint_weight_transform", None)
      moe = self._moe_for_weight(name)
      if callable(checkpoint_transform):
        checkpoint_tensor = checkpoint_transform(checkpoint_tensor)
      elif moe is not None and name.endswith(self._EP_SHARDED_SUFFIXES):
        checkpoint_tensor = moe.ep_slice(checkpoint_tensor)
      if tuple(target.shape) != tuple(checkpoint_tensor.shape):
        raise ValueError(f"MAGI-2 weight {name!r} has shape {tuple(checkpoint_tensor.shape)}, "
                         f"expected {tuple(target.shape)}")
      with torch.no_grad():
        target.copy_(checkpoint_tensor.to(device=target.device, dtype=target.dtype))
      loaded.add(name)

    return loaded

  def _weight_targets(self) -> dict:
    """Return the parameter name -> parameter view map, cached across shard loads.

    :returns: ``state_dict(keep_vars=True)`` cached on the instance.
    """

    targets = getattr(self, "_magi2_weight_targets", None)
    if targets is None:
      targets = self.state_dict(keep_vars=True)
      self._magi2_weight_targets = targets
    return targets

  def finalize_loading(self, loaded: set) -> None:
    """Apply the released router-bias rule and fail closed on missing tensors.

    :param loaded: Names returned by every :meth:`load_weights` call.
    :raises ValueError: When a parameter was not loaded.
    """

    if (os.environ.get("MAGI2_ROUTER_BIAS_SOURCE") or "ema").strip().lower() != "main":
      for module_name, module in self.named_modules():
        if not isinstance(module, Magi2MultiHeadMoE):
          continue
        module.router.expert_bias.copy_(module.router.expert_bias_ema)
        loaded.add(f"{module_name}.router.expert_bias")

    missing = set(self._weight_targets()) - loaded
    if missing:
      raise ValueError(
        f"MAGI-2 preview checkpoint is missing {len(missing)} weights: {sorted(missing)[:8]}")
    self._magi2_weight_targets = None
