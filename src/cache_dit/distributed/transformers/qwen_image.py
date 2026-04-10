import functools
from typing import Dict, List, Optional, Tuple

import diffusers
import torch
from diffusers import QwenImageTransformer2DModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_qwenimage import (
  Attention,
  QwenDoubleStreamAttnProcessor2_0,
  QwenImageTransformerBlock,
  apply_rotary_emb_qwen,
)
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...attention import _dispatch_attention_fn
from ...distributed import (
  _All2AllComm,
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


@ContextParallelismPlannerRegister.register("QwenImage")
class QwenImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    # NOTE: Set it as False to use custom CP plan defined here.
    self._cp_planner_preferred_native_diffusers = False

    if parallelism_config.ulysses_async:
      QwenDoubleStreamAttnProcessor2_0.__call__ = __patch_qwen_attn_processor__

    if transformer is not None and self._cp_planner_preferred_native_diffusers:
      assert isinstance(transformer, QwenImageTransformer2DModel
                        ), "Transformer must be an instance of QwenImageTransformer2DModel"
      if hasattr(transformer, "_cp_plan"):
        if transformer._cp_plan is not None:
          return transformer._cp_plan

    if diffusers.__version__ <= "0.36.0":
      _cp_plan = {
        "": {
          "hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "encoder_hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "encoder_hidden_states_mask":
          _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        "pos_embed": {
          0: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
          1: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      }
    else:
      # Make CP plan compatible with https://github.com/huggingface/diffusers/pull/12702
      _cp_plan = {
        "transformer_blocks.0": {
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

    zero_cond_t = getattr(transformer, "zero_cond_t", False)
    if zero_cond_t:
      # modulate_index: [b, l=seq_len], Qwen-Image-Edit-2511
      _cp_plan.update({
        "transformer_blocks.*": {
          "modulate_index": _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        }
      })

    return _cp_plan


# Implements async Ulysses communication for Attention module when context parallelism
# is enabled with Ulysses degree > 1. The async communication allows overlapping
# communication with computation for better performance.
def _async_ulysses_attn_qwen(
  self: QwenDoubleStreamAttnProcessor2_0,
  attn: Attention,
  hidden_states: torch.FloatTensor,  # Image stream
  encoder_hidden_states: torch.FloatTensor = None,  # Text stream
  encoder_hidden_states_mask: torch.FloatTensor = None,
  attention_mask: Optional[torch.FloatTensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
  if encoder_hidden_states is None:
    raise ValueError(
      "QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

  cp_config = getattr(self, "_cp_config", None)
  if cp_config is None:
    raise RuntimeError(
      "QwenDoubleStreamAttnProcessor2_0 is missing _cp_config during async Ulysses attention.")

  seq_txt = encoder_hidden_states.shape[1]

  img_value = attn.to_v(hidden_states)
  txt_value = attn.add_v_proj(encoder_hidden_states)
  img_value = img_value.unflatten(-1, (attn.heads, -1))
  txt_value = txt_value.unflatten(-1, (attn.heads, -1))
  joint_value = torch.cat([txt_value, img_value], dim=1)

  comm = _All2AllComm(cp_config).init_meta(joint_value)

  # Async all to all for value
  joint_value_wait = comm.send_v(joint_value)

  # Compute QKV for image stream (sample projections)
  img_query = attn.to_q(hidden_states)
  # Compute QKV for text stream (context projections)
  txt_query = attn.add_q_proj(encoder_hidden_states)
  # Reshape for multi-head attention
  img_query = img_query.unflatten(-1, (attn.heads, -1))
  txt_query = txt_query.unflatten(-1, (attn.heads, -1))
  # Apply QK normalization
  if attn.norm_q is not None:
    img_query = attn.norm_q(img_query)
  if attn.norm_added_q is not None:
    txt_query = attn.norm_added_q(txt_query)
  # Apply RoPE
  if image_rotary_emb is not None:
    img_freqs, txt_freqs = image_rotary_emb
    img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
    txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
  # Concatenate for joint attention
  # Order: [text, image]
  joint_query = torch.cat([txt_query, img_query], dim=1)

  # Async all to all for query
  joint_query_wait = comm.send_q(joint_query)

  img_key = attn.to_k(hidden_states)
  txt_key = attn.add_k_proj(encoder_hidden_states)
  img_key = img_key.unflatten(-1, (attn.heads, -1))
  txt_key = txt_key.unflatten(-1, (attn.heads, -1))
  if attn.norm_k is not None:
    img_key = attn.norm_k(img_key)
  if attn.norm_added_k is not None:
    txt_key = attn.norm_added_k(txt_key)
  # Apply RoPE
  if image_rotary_emb is not None:
    img_freqs, txt_freqs = image_rotary_emb
    img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
    txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
  joint_key = torch.cat([txt_key, img_key], dim=1)

  # Async all to all for key
  joint_key_wait = comm.send_k(joint_key)

  # (S_GLOBAL, B, H_LOCAL, D) -> (B, S_GLOBAL, H_LOCAL, D)
  joint_value = joint_value_wait.wait()  # type: torch.Tensor
  joint_query = joint_query_wait.wait()  # type: torch.Tensor
  joint_key = joint_key_wait.wait()  # type: torch.Tensor

  # Compute joint attention
  out = _dispatch_attention_fn(
    joint_query,
    joint_key,
    joint_value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=self._attention_backend,
    cp_config=None,  # set to None to avoid double parallelism
  )  # (B, S_GLOBAL, H_LOCAL, D)

  # TODO: Split output before all to all to apply Async all to all,
  # overlap comm and compute of _to_out and to_add_out projections.
  out_wait = comm.send_o(out)  # (B, S_LOCAL, H_GLOBAL, D)
  joint_hidden_states = out_wait.wait()  # type: torch.Tensor

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


qwen_attn_processor__call__ = QwenDoubleStreamAttnProcessor2_0.__call__


@functools.wraps(qwen_attn_processor__call__)
def __patch_qwen_attn_processor__(
  self: QwenDoubleStreamAttnProcessor2_0,
  attn: Attention,
  hidden_states: torch.FloatTensor,  # Image stream
  encoder_hidden_states: torch.FloatTensor = None,  # Text stream
  encoder_hidden_states_mask: torch.FloatTensor = None,
  attention_mask: Optional[torch.FloatTensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is not None and cp_config.ulysses_degree > 1:
    return _async_ulysses_attn_qwen(
      self,
      attn,
      hidden_states,
      encoder_hidden_states,
      encoder_hidden_states_mask,
      attention_mask,
      image_rotary_emb,
    )
  else:
    return qwen_attn_processor__call__(
      self,
      attn,
      hidden_states,
      encoder_hidden_states,
      encoder_hidden_states_mask,
      attention_mask,
      image_rotary_emb,
    )


@TensorParallelismPlannerRegister.register("QwenImageTransformer2DModel")
class QwenImageTensorParallelismPlanner(TensorParallelismPlanner):

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
    transformer: QwenImageTransformer2DModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    layer_plans = []
    for _, block in transformer.transformer_blocks.named_children():
      assert isinstance(block, QwenImageTransformerBlock)
      shard_div_attr(block.attn, "heads", tp_mesh.size())
      layer_plan = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
        "img_mod.1": ColwiseParallel(output_layouts=Replicate()),
        "img_mlp.net.0.proj": ColwiseParallel(),
        "img_mlp.net.2": RowwiseParallel(),
        "attn.add_q_proj": ColwiseParallel(),
        "attn.add_k_proj": ColwiseParallel(),
        "attn.add_v_proj": ColwiseParallel(),
        "attn.to_add_out": RowwiseParallel(),
        "txt_mod.1": ColwiseParallel(output_layouts=Replicate()),
        "txt_mlp.net.0.proj": ColwiseParallel(),
        "txt_mlp.net.2": RowwiseParallel(),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    return transformer, layer_plans
