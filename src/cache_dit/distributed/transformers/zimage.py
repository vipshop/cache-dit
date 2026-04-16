from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ..async_ulysses import AsyncUlyssesRegistry
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

    if parallelism_config.ulysses_async and transformer is not None:
      AsyncUlyssesRegistry.apply(transformer)

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
