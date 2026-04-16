import functools
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import ZImageTransformer2DModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_z_image import (
  Attention,
  ZSingleStreamAttnProcessor,
)
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...attention import _dispatch_attention_fn
from ...distributed.core import (
  _All2AllComm,
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...platforms import current_platform
from ...logger import init_logger
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("ZImageTransformer2DModel")
class ZImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    # NOTE: Diffusers native CP plan still not supported for ZImageTransformer2DModel
    self._cp_planner_preferred_native_diffusers = False

    if transformer is not None and self._cp_planner_preferred_native_diffusers:
      assert isinstance(
        transformer,
        ZImageTransformer2DModel), "Transformer must be an instance of ZImageTransformer2DModel"
      if hasattr(transformer, "_cp_plan"):
        if transformer._cp_plan is not None:
          return transformer._cp_plan

    if parallelism_config.ulysses_async:
      ZSingleStreamAttnProcessor.__call__ = __patch_zimage_attn_processor__

    n_noise_refiner_layers = len(transformer.noise_refiner)  # 2
    n_context_refiner_layers = len(transformer.context_refiner)  # 2
    n_layers = len(transformer.layers)  # 30
    # controlnet layer idx: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    # num_controlnet_samples = len(transformer.layers) // 2  # 15
    has_controlnet = parallelism_config._has_controlnet
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
      # Special cp plan for ZImageTransformer2DModel with ZImageControlNetModel
      logger.warning("Using special context parallelism plan for ZImageTransformer2DModel "
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


# Implements async Ulysses communication for Attention module when context parallelism
# is enabled with Ulysses degree > 1. The async communication allows overlapping
# communication with computation for better performance.
def _async_ulysses_attn_zimage(
  self: ZSingleStreamAttnProcessor,
  attn: Attention,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  freqs_cis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is None:
    raise RuntimeError(
      "ZSingleStreamAttnProcessor is missing _cp_config during async Ulysses attention.")

  # Apply RoPE
  def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(current_platform.device_type, enabled=False):
      x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
      freqs_cis = freqs_cis.unsqueeze(2)
      x_out = torch.view_as_real(x * freqs_cis).flatten(3)
      return x_out.type_as(x_in)  # todo

  dtype = hidden_states.dtype
  query = attn.to_q(hidden_states)  # type: torch.Tensor
  query = query.unflatten(-1, (attn.heads, -1))
  if attn.norm_q is not None:  # Apply Norms
    query = attn.norm_q(query)
  if freqs_cis is not None:  # Apply RoPE
    query = apply_rotary_emb(query, freqs_cis)

  comm = _All2AllComm(cp_config)

  # Async all to all for query
  query_wait = comm.send_q(query)

  key = attn.to_k(hidden_states)  # type: torch.Tensor
  key = key.unflatten(-1, (attn.heads, -1))
  if attn.norm_k is not None:  # Apply Norms
    key = attn.norm_k(key)
  if freqs_cis is not None:  # Apply RoPE
    key = apply_rotary_emb(key, freqs_cis)

  # Async all to all for key
  key_wait = comm.send_k(key)

  value = attn.to_v(hidden_states)  # type: torch.Tensor
  value = value.unflatten(-1, (attn.heads, -1))

  # Async all to all for value
  value_wait = comm.send_v(value)

  # Ensure the query, key, value are ready
  query = query_wait.wait()
  key = key_wait.wait()
  value = value_wait.wait()

  # Cast to correct dtype
  query, key = query.to(dtype), key.to(dtype)

  # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
  if attention_mask is not None and attention_mask.ndim == 2:
    attention_mask = attention_mask[:, None, None, :]

  # Compute joint attention
  out = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=self._attention_backend,
    cp_config=None,  # set to None to avoid double parallelism
  )  # (B, S_GLOBAL, H_LOCAL, D)

  out_wait = comm.send_o(out)  # (B, S_LOCAL, H_GLOBAL, D)
  hidden_states = out_wait.wait()  # type: torch.Tensor

  # Reshape back
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.to(dtype)

  output = attn.to_out[0](hidden_states)
  if len(attn.to_out) > 1:  # dropout
    output = attn.to_out[1](output)

  return output


zimage_attn_processor__call__ = ZSingleStreamAttnProcessor.__call__


@functools.wraps(zimage_attn_processor__call__)
def __patch_zimage_attn_processor__(
  self: ZSingleStreamAttnProcessor,
  attn: Attention,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  freqs_cis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is not None and cp_config.ulysses_degree > 1:
    return _async_ulysses_attn_zimage(
      self,
      attn,
      hidden_states,
      encoder_hidden_states,
      attention_mask,
      freqs_cis,
    )
  else:
    return zimage_attn_processor__call__(
      self,
      attn,
      hidden_states,
      encoder_hidden_states,
      attention_mask,
      freqs_cis,
    )


@TensorParallelismPlannerRegister.register("Lumina2")
@TensorParallelismPlannerRegister.register("ZImage")
class ZImageTensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    transformer, layer_plans = self.parallelize_transformer(
      transformer=transformer,
      tp_mesh=tp_mesh,
    )

    return transformer, layer_plans

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    class_name = transformer.__class__.__name__

    attn_mod_name = "attention" if class_name.startswith("ZImage") else "attn"
    ff_linear_name = "w" if class_name.startswith("ZImage") else "linear_"

    def tp_shard_block(block, tp_size):
      attn = getattr(block, attn_mod_name)
      shard_div_attr(attn, "heads", tp_size)
      layer_plan = {
        f"{attn_mod_name}.to_q": ColwiseParallel(),
        f"{attn_mod_name}.to_k": ColwiseParallel(),
        f"{attn_mod_name}.to_v": ColwiseParallel(),
        f"{attn_mod_name}.to_out.0": RowwiseParallel(),
        f"feed_forward.{ff_linear_name}1": ColwiseParallel(),
        f"feed_forward.{ff_linear_name}3": ColwiseParallel(),
        f"feed_forward.{ff_linear_name}2": RowwiseParallel(),
        # saving more memory at the cost of more communication
        # "adaLN_modulation.0": ColwiseParallel(output_layouts=Replicate()),
      }

      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      return layer_plan

    tp_size = tp_mesh.get_group().size()
    layer_plans = []
    for _, block in transformer.noise_refiner.named_children():
      layer_plans.append(tp_shard_block(block, tp_size))
    for _, block in transformer.context_refiner.named_children():
      layer_plans.append(tp_shard_block(block, tp_size))
    for _, block in transformer.layers.named_children():
      layer_plans.append(tp_shard_block(block, tp_size))

    return transformer, layer_plans
