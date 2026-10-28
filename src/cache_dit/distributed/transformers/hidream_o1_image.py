from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...logger import init_logger
from ..config import ParallelismConfig
from ..core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("HiDreamO1ImageTransformer2DModel")
class HiDreamO1ImageContextParallelismPlanner(ContextParallelismPlanner):
  """Hook-based context parallelism for HiDream-O1-Image.

  The model keeps text, the timestep token and the image patches in **one** joint sequence, so the
  sequence is split at the denoising stack and everything outside it stays replicated:

  * `layers.0` splits `hidden_states` on dim 1. cache-dit's Fn-residual replay reads the split hook
    of `layers[0]`, so the input split has to live on the first block.
  * `rope_split` output-splits the packed mRoPE table `[B, S, 2 * head_dim]` with the same
    partition. The table is built on the full `position_ids` (which is therefore not split), and the
    shrink follows the framework sharder so `--ulysses-anything` stays consistent.
  * `sequence_gather` gathers dim 1 back before `norm` / `final_layer2`, so the pipeline can slice
    the image rows out of the full-length prediction.

  **The attention mask is not split and not permuted.** Ulysses all-to-all turns Q/K/V into
  `[B, S_full, H_local, D]`, so the dense additive `[B, 1, S, S]` mask still matches. Unlike models
  whose joint sequence is `[text_all, image_all]` built from two separately split tensors, this
  model's chunks are contiguous slices of a single sequence, so rank-concatenation restores the
  original order and the required permutation is the identity.

  Ring attention is rejected: every ring step holds only a local K/V chunk and requests LSE, which
  the native backend refuses together with `attn_mask`, and the global mask would have to be rolled
  per step. Ring also needs equally sized chunks, while the joint length is rarely divisible by the
  CP size (`--ulysses-anything` only covers Ulysses).
  """

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert parallelism_config is not None, "ParallelismConfig must be provided."
    ring_size = parallelism_config.ring_size
    if ring_size is not None and ring_size > 1:
      raise NotImplementedError(
        "HiDream-O1-Image context parallelism does not support Ring attention: the mixed "
        "causal/full attention mask is a dense [B, 1, S, S] tensor that a ring step cannot consume, "
        "and Ring needs equally sized chunks while the joint length is rarely divisible by the CP "
        "size. Use `--parallel ulysses` (add `--ulysses-anything` when the length is not divisible)."
      )

    _cp_plan: _ContextParallelModelPlan = {
      "rope_split": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
      },
      "layers.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      "sequence_gather": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    logger.debug(f"Generated HiDream-O1-Image CP plan: {_cp_plan}")
    return _cp_plan


@TensorParallelismPlannerRegister.register("HiDreamO1ImageTransformer2DModel")
class HiDreamO1ImageTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor-parallel planner for HiDream-O1-Image.

  Each block is a plain Qwen3-VL decoder layer: separate `q_proj` / `k_proj` / `v_proj` / `o_proj`
  and a gated FFN (`gate_proj`, `up_proj`, `down_proj`). Nothing is packed, so no weight
  rearrangement is needed.

  * Q/K/V are `ColwiseParallel` and both `num_heads` and `num_kv_heads` are divided by `tp_size`
    (32/8 heads divide by 2 and 4). The attention module reshapes with those counts and widens GQA
    with `num_heads // num_kv_heads`, which is unchanged by the division, so the repeat factor stays
    correct. `head_dim` must **not** be divided: the per-head norms operate on it and it never
    crosses a shard boundary.
  * `o_proj` and `down_proj` are `RowwiseParallel` and all-reduce the partial sums.
  * `gate_proj` / `up_proj` are `ColwiseParallel`.

  Everything outside `layers` (token embedding, timestep MLP, patch embed, output head, norms)
  stays replicated: the residual stream is full-length after each all-reduce.
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

  @staticmethod
  def layer_plan() -> Dict[str, ParallelStyle]:
    """Per-block parallel style map, identical for every decoder layer.

    :returns: Mapping of submodule path to parallel style.
    """
    return {
      "self_attn.q_proj": ColwiseParallel(),
      "self_attn.k_proj": ColwiseParallel(),
      "self_attn.v_proj": ColwiseParallel(),
      "self_attn.o_proj": RowwiseParallel(),
      "mlp.gate_proj": ColwiseParallel(),
      "mlp.up_proj": ColwiseParallel(),
      "mlp.down_proj": RowwiseParallel(),
    }

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_size = tp_mesh.size()
    layer_plans: List[Dict[str, ParallelStyle]] = []
    for block in transformer.layers:
      assert isinstance(block, nn.Module), "HiDream-O1-Image layers must hold `nn.Module` blocks."
      # Both counts drive the head reshape inside the attention module; `head_dim` does not.
      shard_div_attr(block.self_attn, "num_heads", tp_size)
      shard_div_attr(block.self_attn, "num_kv_heads", tp_size)
      layer_plan = self.layer_plan()
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    return transformer, layer_plans
