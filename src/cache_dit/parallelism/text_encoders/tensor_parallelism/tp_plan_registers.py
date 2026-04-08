import torch
import logging
from abc import abstractmethod
from typing import Dict, List, Tuple
from torch.distributed import init_device_mesh
from torch.distributed.tensor.parallel import (
  ParallelStyle,
  ColwiseParallel,
  RowwiseParallel,
)
from cache_dit.parallelism.config import ParallelismConfig
from ....logger import init_logger

logger = init_logger(__name__)


class TextEncoderTensorParallelismPlanner:

  def __init__(self):
    # Record the planned parallelization for each layer, which can be used for
    # later quantization or other purposes. For example, the plans for FLUX.2's
    # single block may be like:
    # layer_plan = {
    #     "attn.to_qkv_mlp_proj": ColwiseParallel(),
    #     "attn.to_out": RowwiseParallel(),
    # }
    self.layer_plan_records = {}

  @abstractmethod
  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:  # text_encoder and layer plans
    raise NotImplementedError("apply method must be implemented by subclasses")

  def mesh(self, parallelism_config: ParallelismConfig, **kwargs):
    text_encoder_world_size = parallelism_config.text_encoder_world_size
    device_type = torch.accelerator.current_accelerator().type
    tp_mesh = init_device_mesh(
      device_type=device_type,
      mesh_shape=[text_encoder_world_size],
    )
    return tp_mesh

  def apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> torch.nn.Module:
    text_encoder, layer_plans = self._apply(
      text_encoder=text_encoder,
      parallelism_config=parallelism_config,
      **kwargs,
    )

    # Workaround for case: TP -> FP8 DQ per row, make torch._scaled_mm happy.
    # Avoid error: "RuntimeError: Expected b.stride(0) == 1 to be true, but got false"
    # RowwiseParallel (TP) will cause the layout of the linear weights changedly after
    # '_dispatch_get_local_results_slow_path', Why??? Need further investigation.
    self.record_plans(text_encoder, layer_plans)

    return text_encoder

  def record_plans(self, text_encoder: torch.nn.Module, layer_plans: List[Dict[str,
                                                                               ParallelStyle]]):
    # Remove duplicate layer plans to avoid recording the same plan multiple times
    # for different blocks. For example, in FLUX.2, the main block and single block
    # have the same parallelization plan, we only need to record it once.
    unique_layer_plans = []
    unique_plan_ids = []
    seen_plans = set()
    for layer_plan in layer_plans:
      plan_id = hash(frozenset(layer_plan.keys()))
      if plan_id not in seen_plans:
        seen_plans.add(plan_id)
        unique_layer_plans.append(layer_plan)
        unique_plan_ids.append(plan_id)

    # Asssign all the keys in recorded layer plan with the same parallel style, for
    # later quantization or other purposes. For example, if the layer_plan is:
    # {"attn.to_qkv_mlp_proj": ColwiseParallel(), "attn.to_q": ColwiseParallel(), ...},
    # then we will assign:
    # text_encoder._rowwise_layers = [attn.to_out, attn.to_out.0, ff.linear_out, ...]
    # text_encoder._colwise_layers = [attn.to_qkv_mlp_proj, attn.to_q, attn.to_k, ...]
    rowwise_layers = set()
    colwise_layers = set()
    for plan_id, layer_plan in zip(unique_plan_ids, unique_layer_plans):
      self.layer_plan_records[plan_id] = layer_plan
      for layer, parallel_style in layer_plan.items():
        if isinstance(parallel_style, RowwiseParallel):
          rowwise_layers.add(layer)
        elif isinstance(parallel_style, ColwiseParallel):
          colwise_layers.add(layer)
    text_encoder._rowwise_layers = list(rowwise_layers)
    text_encoder._colwise_layers = list(colwise_layers)


class TextEncoderTensorParallelismPlannerRegister:
  _text_encoder_tp_planner_registry: Dict[str, TextEncoderTensorParallelismPlanner] = {}

  @classmethod
  def register(cls, name: str):

    def decorator(planner_cls: type[TextEncoderTensorParallelismPlanner]):
      assert (name not in cls._text_encoder_tp_planner_registry
              ), f"TextEncoderTensorParallelismPlanner with name {name} is already registered."
      if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Registering TextEncoderTensorParallelismPlanner: {name}")
      cls._text_encoder_tp_planner_registry[name] = planner_cls
      return planner_cls

    return decorator

  @classmethod
  def get_planner(cls,
                  text_encoder: str | torch.nn.Module) -> type[TextEncoderTensorParallelismPlanner]:
    if isinstance(text_encoder, torch.nn.Module):
      name = text_encoder.__class__.__name__
    else:
      name = text_encoder
    planner_cls = None
    for planner_name in cls._text_encoder_tp_planner_registry:
      if name.startswith(planner_name):
        planner_cls = cls._text_encoder_tp_planner_registry.get(planner_name)
        break
    if planner_cls is None:
      raise ValueError(f"No planner registered under name: {name}")
    return planner_cls

  @classmethod
  def supported_planners(cls, ) -> tuple[int, list[str]]:
    val_planners = cls._text_encoder_tp_planner_registry.keys()
    return len(val_planners), [p for p in val_planners]
