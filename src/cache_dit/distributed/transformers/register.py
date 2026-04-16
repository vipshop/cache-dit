import logging
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin
from torch.distributed import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel

from ...distributed.core import _ContextParallelModelPlan
from ...logger import init_logger
from ..config import ParallelismConfig

logger = init_logger(__name__)

__all__ = [
  "ContextParallelismPlanner",
  "ContextParallelismPlannerRegister",
  "TensorParallelismPlanner",
  "TensorParallelismPlannerRegister",
  "ParallelismConfig",
]


class ContextParallelismPlanner:

  def apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    """Apply a context-parallel planner to a transformer instance.

    :param transformer: Transformer model that should receive the generated CP plan.
    :param parallelism_config: Parallelism configuration used while generating the plan.
    :param kwargs: Extra planner-specific options.
    :returns: An internal `_ContextParallelModelPlan` describing split/gather hooks.
    """
    assert (transformer
            is not None), "Transformer model must be provided to apply context parallelism."
    assert (parallelism_config
            is not None), "ParallelismConfig must be provided to apply context parallelism."
    cp_plan = self._apply(
      transformer=transformer,
      parallelism_config=parallelism_config,
      **kwargs,
    )
    transformer._cp_plan = cp_plan
    cls_name = transformer.__class__.__name__
    if logger.isEnabledFor(logging.DEBUG):
      logger.debug(f"Generated CP plan: {cp_plan}")
    if getattr(parallelism_config, "ulysses_async", False):
      logger.info(f"Async Ulysses Attention is enabled for {cls_name}.")
    return cp_plan

  @abstractmethod
  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    raise NotImplementedError("apply method must be implemented by subclasses")


class ContextParallelismPlannerRegister:
  _cp_planner_registry: dict[str, type[ContextParallelismPlanner]] = {}

  @classmethod
  def register(cls, name: str):

    def decorator(planner_cls: type[ContextParallelismPlanner]):
      assert (name not in cls._cp_planner_registry
              ), f"ContextParallelismPlanner with name {name} is already registered."
      if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Registering ContextParallelismPlanner: {name}")
      cls._cp_planner_registry[name] = planner_cls
      return planner_cls

    return decorator

  @classmethod
  def get_planner(
    cls,
    transformer: str | torch.nn.Module | ModelMixin,
  ) -> type[ContextParallelismPlanner]:
    if isinstance(transformer, (torch.nn.Module, ModelMixin)):
      name = transformer.__class__.__name__
    else:
      name = transformer
    planner_cls = None
    for planner_name in cls._cp_planner_registry:
      if name.startswith(planner_name):
        planner_cls = cls._cp_planner_registry.get(planner_name)
        break
    if planner_cls is None:
      raise ValueError(f"No planner registered under name: {name}")
    return planner_cls

  @classmethod
  def supported_planners(cls) -> tuple[int, list[str]]:
    val_planners = cls._cp_planner_registry.keys()
    return len(val_planners), [planner for planner in val_planners]


class TensorParallelismPlanner:

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
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    raise NotImplementedError("_apply method must be implemented by subclasses")

  def apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> torch.nn.Module:
    transformer, layer_plans = self._apply(
      transformer=transformer,
      parallelism_config=parallelism_config,
      **kwargs,
    )

    # Workaround for case: TP -> FP8 DQ per row, make torch._scaled_mm happy.
    # Avoid error: "RuntimeError: Expected b.stride(0) == 1 to be true, but got false"
    # RowwiseParallel (TP) will cause the layout of the linear weights changedly after
    # '_dispatch_get_local_results_slow_path', Why??? Need further investigation.
    self.record_plans(transformer, layer_plans)

    return transformer

  def mesh(self, parallelism_config: ParallelismConfig, **kwargs):
    assert (parallelism_config.tp_enabled()
            ), "tp_size must be set and greater than 1 for tensor parallelism"
    if parallelism_config.hybrid_enabled() and parallelism_config._tp_mesh is not None:
      return parallelism_config._tp_mesh

    device_type = torch.accelerator.current_accelerator().type
    return init_device_mesh(
      device_type=device_type,
      mesh_shape=[parallelism_config.tp_size],
    )

  def record_plans(self, transformer: torch.nn.Module, layer_plans: List[Dict[str, ParallelStyle]]):
    unique_layer_plans = []
    unique_plan_ids = []
    seen_plans = set()
    for layer_plan in layer_plans:
      plan_id = hash(frozenset(layer_plan.keys()))
      if plan_id not in seen_plans:
        seen_plans.add(plan_id)
        unique_layer_plans.append(layer_plan)
        unique_plan_ids.append(plan_id)

    rowwise_layers = set()
    colwise_layers = set()
    for plan_id, layer_plan in zip(unique_plan_ids, unique_layer_plans):
      self.layer_plan_records[plan_id] = layer_plan
      for layer, parallel_style in layer_plan.items():
        if isinstance(parallel_style, RowwiseParallel):
          rowwise_layers.add(layer)
        elif isinstance(parallel_style, ColwiseParallel):
          colwise_layers.add(layer)
    transformer._rowwise_layers = list(rowwise_layers)
    transformer._colwise_layers = list(colwise_layers)


class TensorParallelismPlannerRegister:
  _tp_planner_registry: Dict[str, type[TensorParallelismPlanner]] = {}

  @classmethod
  def register(cls, name: str):

    def decorator(planner_cls: type[TensorParallelismPlanner]):
      assert (name not in cls._tp_planner_registry
              ), f"TensorParallelismPlanner with name {name} is already registered."
      if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Registering TensorParallelismPlanner: {name}")
      cls._tp_planner_registry[name] = planner_cls
      return planner_cls

    return decorator

  @classmethod
  def get_planner(cls, transformer: str | torch.nn.Module) -> type[TensorParallelismPlanner]:
    if isinstance(transformer, torch.nn.Module):
      name = transformer.__class__.__name__
    else:
      name = transformer
    planner_cls = None
    for planner_name in cls._tp_planner_registry:
      if name.startswith(planner_name):
        planner_cls = cls._tp_planner_registry.get(planner_name)
        break
    if planner_cls is None:
      raise ValueError(f"No planner registered under name: {name}")
    return planner_cls

  @classmethod
  def supported_planners(cls) -> tuple[int, list[str]]:
    val_planners = cls._tp_planner_registry.keys()
    return len(val_planners), [planner for planner in val_planners]
