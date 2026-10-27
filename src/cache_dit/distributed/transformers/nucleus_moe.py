"""NucleusMoE-Image tensor-parallel planner.

`NucleusMoEImageTransformer2DModel` is a single-stream DiT whose attention is *asymmetric*: the
image tokens are the only queries, and every image query attends to the concatenation of the image
K/V with the text K/V (`torch.cat([img_key, txt_key])` inside `NucleusMoEAttnProcessor2_0`).

**Context parallelism is deliberately not offered.** The stack is 3 dense + 29 expert-choice MoE
blocks, and `NucleusMoELayer` derives its per-expert budget from the sequence length it receives
(`ceil(capacity_factor * S / num_experts)`): on a sharded sequence every expert would pick its
top-C tokens from its own rank's pool, so the selected token set — and with it the residual — is a
different function, not a numerically equivalent one. Making it exact needs the router logits
gathered across ranks before the top-C selection, which is a separate piece of work. Tensor
parallelism does not have this problem: it shards weights, so the routing input is unchanged.
"""

from typing import Dict, List, Tuple

import torch
from diffusers.models.transformers.transformer_nucleusmoe_image import (
  NucleusMoEImageTransformer2DModel, )
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import (
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)


@TensorParallelismPlannerRegister.register("NucleusMoEImageTransformer2DModel")
class NucleusMoETensorParallelismPlanner(TensorParallelismPlanner):
  r"""Tensor-parallel planner for NucleusMoE-Image.

  * Attention is GQA (`heads=16`, 4 KV heads) with a *joint* KV: the text K/V comes from
    `attn.add_k_proj` / `attn.add_v_proj` and is concatenated with the image K/V before the
    attention. Both halves have to be sharded on the same column grid, otherwise the concatenation
    mixes widths. `heads`, `inner_dim` and `inner_kv_dim` are divided by `tp_size` because the
    processor derives `head_dim` and the KV head count from them.
  * `encoder_proj` (per block) and `img_in` / `proj_out` (transformer) keep a replicated output:
    their consumers address the full feature width.
  * Dense blocks run a SwiGLU `FeedForward` and MoE blocks a `shared_expert` of the same shape, so
    the fused `(up, gate)` projection is rearranged before the column shard (the same interleave
    the MiniMax-H3 plan uses), then `net.0.proj` is column- and `net.2` row-parallel.
  * The routed experts (`SwiGLUExperts.gate_up_proj` / `down_proj`) stay **replicated**: they are
    raw 3-D `nn.Parameter` tables consumed by `F.grouped_mm`, which has no DTensor strategy, and
    sharding `down_proj` on its reduction axis would need an all-reduce per expert. They are
    ~30.6 GiB of the 51.7 GiB checkpoint, so TP2 saves roughly 10 GiB rather than half the model.
  * `img_mod` (`chunk(4)`), the MoE router `gate` (`cat([temb, x])` input) and `txt_norm` stay
    replicated.
  """

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
    transformer: NucleusMoEImageTransformer2DModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    from .ovis_image import rearrange_ffn_0_swiglu_proj_weight

    tp_size = tp_mesh.size()
    heads = transformer.config.num_attention_heads
    kv_heads = transformer.config.num_key_value_heads or heads
    if heads % tp_size != 0 or kv_heads % tp_size != 0:
      raise ValueError(f"[TP] Unsupported tp_size={tp_size}: {heads} query heads and {kv_heads} "
                       f"KV heads must both be divisible by tp_size.")

    layer_plans: List[Dict[str, ParallelStyle]] = []
    for block in transformer.transformer_blocks:
      attention_plan: Dict[str, ParallelStyle] = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.add_k_proj": ColwiseParallel(),
        "attn.add_v_proj": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
        # Downstream `add_k_proj` consumes the full width as its input.
        "encoder_proj": ColwiseParallel(output_layouts=Replicate()),
      }
      self.rearrange_swiglu_proj(block, tp_size, rearrange_ffn_0_swiglu_proj_weight)
      if block.moe_enabled:
        attention_plan["img_mlp.shared_expert.net.0.proj"] = ColwiseParallel()
        attention_plan["img_mlp.shared_expert.net.2"] = RowwiseParallel()
      else:
        attention_plan["img_mlp.net.0.proj"] = ColwiseParallel()
        attention_plan["img_mlp.net.2"] = RowwiseParallel()

      # The processor re-derives `head_dim` and the KV head count from these three attributes.
      shard_div_attr(block.attn, "heads", tp_size)
      shard_div_attr(block.attn, "inner_dim", tp_size)
      shard_div_attr(block.attn, "inner_kv_dim", tp_size)

      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=attention_plan)
      layer_plans.append(attention_plan)

    transformer_plan: Dict[str, ParallelStyle] = {
      "img_in": ColwiseParallel(output_layouts=Replicate()),
      "proj_out": ColwiseParallel(output_layouts=Replicate()),
    }
    parallelize_module(module=transformer, device_mesh=tp_mesh, parallelize_plan=transformer_plan)
    layer_plans.append(transformer_plan)

    local_q_rows = transformer.transformer_blocks[0].attn.to_q.weight.to_local().shape[0]
    if local_q_rows != (heads // tp_size) * transformer.config.attention_head_dim:
      raise ValueError(f"Sharded Q projection has {local_q_rows} rows per rank, expected "
                       f"{heads // tp_size} heads x {transformer.config.attention_head_dim} "
                       "channels.")
    return transformer, layer_plans

  @staticmethod
  def rearrange_swiglu_proj(block: torch.nn.Module, tp_size: int, rearrange) -> None:
    """Interleave the fused `(up, gate)` projection of a block's SwiGLU FFN.

    :param block: NucleusMoE block whose FFN is rearranged in place.
    :param tp_size: TP world size.
    :param rearrange: The shared `(proj, tp_size)` interleave helper.
    """
    ff = block.img_mlp.shared_expert if block.moe_enabled else block.img_mlp
    proj = ff.net[0].proj
    if proj.weight.shape[0] % (2 * tp_size) != 0:
      raise ValueError(f"[TP] Unsupported tp_size={tp_size}: the fused SwiGLU output dimension "
                       f"{proj.weight.shape[0]} must be divisible by 2 * tp_size.")
    rearrange(proj, tp_size)
