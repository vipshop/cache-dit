import functools
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import (
  Attention,
  AttnProcessor2_0,
)  # sdpa
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.pixart_transformer_2d import PixArtTransformer2DModel
from diffusers.utils import deprecate
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...attention import _dispatch_attention_fn
from ...distributed import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
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


@ContextParallelismPlannerRegister.register("PixArt")
class PixArtContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert isinstance(
      transformer,
      PixArtTransformer2DModel), "Transformer must be an instance of PixArtTransformer2DModel"

    self._cp_planner_preferred_native_diffusers = False

    if transformer is not None and self._cp_planner_preferred_native_diffusers:
      if hasattr(transformer, "_cp_plan"):
        if transformer._cp_plan is not None:
          return transformer._cp_plan

    # Apply monkey patch to fix attention mask preparation at class level
    Attention.prepare_attention_mask = __patch_Attention_prepare_attention_mask__
    AttnProcessor2_0.__call__ = __patch_AttnProcessor2_0__call__
    if not hasattr(AttnProcessor2_0, "_parallel_config"):
      AttnProcessor2_0._parallel_config = None
    if not hasattr(AttnProcessor2_0, "_attention_backend"):
      AttnProcessor2_0._attention_backend = None

    # Otherwise, use the custom CP plan defined here, this maybe
    # a little different from the native diffusers implementation
    # for some models.

    _cp_plan = {
      # Pattern of transformer_blocks.0, split_output=False:
      #     un-split input -> split -> to_qkv/...
      #     -> all2all
      #     -> attn (local head, full seqlen)
      #     -> all2all
      #     -> splited output
      #     (only split hidden_states, not encoder_hidden_states)
      "transformer_blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      # Pattern of the all blocks, split_output=False:
      #     un-split input -> split -> to_qkv/...
      #     -> all2all
      #     -> attn (local head, full seqlen)
      #     -> all2all
      #     -> splited output
      #    (only split encoder_hidden_states, not hidden_states.
      #    hidden_states has been automatically split in previous
      #    block by all2all comm op after attn)
      # The `encoder_hidden_states` will [NOT] be changed after each block forward,
      # so we need to split it at [ALL] block by the inserted split hook.
      "transformer_blocks.*": {
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      # Then, the final proj_out will gather the splited output.
      #     splited input (previous splited output)
      #     -> all gather
      #     -> un-split output
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@functools.wraps(Attention.prepare_attention_mask)
def __patch_Attention_prepare_attention_mask__(
  self: Attention,
  attention_mask: torch.Tensor,
  target_length: int,
  batch_size: int,
  out_dim: int = 3,
  # NOTE(DefTruth): Allow specifying head_size for CP
  head_size: Optional[int] = None,
) -> torch.Tensor:
  if head_size is None:
    head_size = self.heads
  if attention_mask is None:
    return attention_mask

  current_length: int = attention_mask.shape[-1]
  if current_length != target_length:
    if attention_mask.device.type == "mps":
      # HACK: MPS: Does not support padding by greater than dimension of input tensor.
      # Instead, we can manually construct the padding tensor.
      padding_shape = (
        attention_mask.shape[0],
        attention_mask.shape[1],
        target_length,
      )
      padding = torch.zeros(
        padding_shape,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
      )
      attention_mask = torch.cat([attention_mask, padding], dim=2)
    else:
      # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
      #       we want to instead pad by (0, remaining_length), where remaining_length is:
      #       remaining_length: int = target_length - current_length
      # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
      attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

  if out_dim == 3:
    if attention_mask.shape[0] < batch_size * head_size:
      attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
  elif out_dim == 4:
    attention_mask = attention_mask.unsqueeze(1)
    attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

  return attention_mask


@functools.wraps(AttnProcessor2_0.__call__)
def __patch_AttnProcessor2_0__call__(
  self: AttnProcessor2_0,
  attn: Attention,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  temb: Optional[torch.Tensor] = None,
  *args,
  **kwargs,
) -> torch.Tensor:
  if len(args) > 0 or kwargs.get("scale", None) is not None:
    deprecation_message = (
      "The `scale` argument is deprecated and will be ignored. Please remove it, "
      "as passing it will raise an error in the future. `scale` should be passed "
      "through the underlying pipeline component, for example via `cross_attention_kwargs`.")
    deprecate("scale", "1.0.0", deprecation_message)

  residual = hidden_states
  if attn.spatial_norm is not None:
    hidden_states = attn.spatial_norm(hidden_states, temb)

  input_ndim = hidden_states.ndim

  if input_ndim == 4:
    batch_size, channel, height, width = hidden_states.shape
    hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

  batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else
                                    encoder_hidden_states.shape)

  if attention_mask is not None:
    if self._cp_config is None:
      attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
      attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
    else:
      # NOTE(DefTruth): Fix attention mask preparation for context parallelism.
      # Please note that in context parallelism, the sequence_length is the local
      # sequence length on each rank. So we need to adjust the target_length
      # accordingly. The head_size is also adjusted based on the world size
      # in order to make sdpa work correctly, otherwise, the sdpa op will raise
      # error due to the mismatch between attention_mask shape and expected shape.
      cp_config = self._cp_config
      if cp_config is not None and cp_config._world_size > 1:
        head_size = attn.heads // cp_config._world_size
        attention_mask = attn.prepare_attention_mask(
          attention_mask,
          sequence_length * cp_config._world_size,
          batch_size,
          3,
          head_size,
        )
        attention_mask = attention_mask.view(batch_size, head_size, -1, attention_mask.shape[-1])

  if attn.group_norm is not None:
    hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

  query = attn.to_q(hidden_states)

  if encoder_hidden_states is None:
    encoder_hidden_states = hidden_states
  elif attn.norm_cross:
    encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

  key = attn.to_k(encoder_hidden_states)
  value = attn.to_v(encoder_hidden_states)

  inner_dim = key.shape[-1]
  head_dim = inner_dim // attn.heads

  # NOTE(DefTruth): no transpose now
  query = query.view(batch_size, -1, attn.heads, head_dim)
  key = key.view(batch_size, -1, attn.heads, head_dim)
  value = value.view(batch_size, -1, attn.heads, head_dim)

  if attn.norm_q is not None:
    query = attn.norm_q(query)
  if attn.norm_k is not None:
    key = attn.norm_k(key)

  # NOTE(DefTruth): Use the dispatch_attention_fn to support different backends
  hidden_states = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=getattr(self, "_attention_backend", None),
    cp_config=getattr(self, "_cp_config", None),
  )
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.to(query.dtype)

  # linear proj
  hidden_states = attn.to_out[0](hidden_states)
  # dropout
  hidden_states = attn.to_out[1](hidden_states)

  if input_ndim == 4:
    hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

  if attn.residual_connection:
    hidden_states = hidden_states + residual

  hidden_states = hidden_states / attn.rescale_output_factor

  return hidden_states


@TensorParallelismPlannerRegister.register("PixArt")
class PixArtTensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer: nn.Module,
    parallelism_config: ParallelismConfig,
    **_kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    transformer, layer_plans = self.parallelize_transformer(
      transformer=transformer,
      tp_mesh=tp_mesh,
    )

    return transformer, layer_plans

  def parallelize_transformer(
    self,
    transformer: nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    """Parallelize PixArt transformer blocks.

    PixArt uses `BasicTransformerBlock` layers with self-attention, cross-attention, feed-forward,
    and standard normalization submodules.

    :param transformer: Transformer module to process.
    :param tp_mesh: Tensor-parallel device mesh.
    :returns: The parallelized transformer and its per-layer sharding plan.
    """
    layer_plans = []
    for i, block in enumerate(transformer.transformer_blocks):
      # Split attention heads across TP devices
      tp_size = tp_mesh.size()
      shard_div_attr(block.attn1, "heads", tp_size)
      shard_div_attr(block.attn2, "heads", tp_size)

      # Create layer plan for tensor parallelism
      layer_plan = {
        # Self-attention projections (column-wise)
        "attn1.to_q": ColwiseParallel(),
        "attn1.to_k": ColwiseParallel(),
        "attn1.to_v": ColwiseParallel(),
        "attn1.to_out.0": RowwiseParallel(),
        # Cross-attention projections (column-wise)
        "attn2.to_q": ColwiseParallel(),
        "attn2.to_k": ColwiseParallel(),
        "attn2.to_v": ColwiseParallel(),
        "attn2.to_out.0": RowwiseParallel(),
        # Feed-forward network
        "ff.net.0.proj": ColwiseParallel(),
        "ff.net.2": RowwiseParallel(),
      }

      # Apply tensor parallelism to the block
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    return transformer, layer_plans
