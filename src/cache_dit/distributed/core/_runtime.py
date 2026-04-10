"""Adapted from diffusers' parallelism enablement entrypoint for model modules."""

import torch
from typing import Dict, Optional
from diffusers import ModelMixin
from ...logger import init_logger
from ._context_parallel import _apply_context_parallel
from ._diffusers_bridge import (
  _normalize_parallel_config,
  get_diffusers_attention_classes,
  validate_context_parallel_attention_backend,
)
from ._modeling_parallel import (
  _ContextParallelModelPlan,
  _ContextParallelConfig,
)

logger = init_logger(__name__)


def _is_diffusers_parallelism_available() -> bool:
  """Return whether the installed diffusers build exposes CP integration APIs."""

  try:
    from diffusers.models import _modeling_parallel as _diffusers_modeling_parallel  # noqa F401
  except ImportError:
    return False
  return True


# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/modeling_utils.py#L1510
def _enable_context_parallelism(
  model: ModelMixin | torch.nn.Module,  # e.g Transformer
  *,
  config: _ContextParallelConfig,
  cp_plan: Optional[Dict[str, _ContextParallelModelPlan]] = None,
  attn_classes_extra: Optional[tuple] = None,
):
  logger.debug("Dispatch parallelism using the cache-dit internal context parallelism api.")

  if not torch.distributed.is_available() or not torch.distributed.is_initialized():
    raise RuntimeError(
      "torch.distributed must be available and initialized before calling `enable_parallelism`.")

  config = _normalize_parallel_config(config)

  rank = torch.distributed.get_rank()
  world_size = torch.distributed.get_world_size()
  device_type = torch._C._get_accelerator().type
  device_module = torch.get_device_module(device_type)
  device = torch.device(device_type, rank % device_module.device_count())

  validate_context_parallel_attention_backend(
    model,
    config,
    attn_classes_extra=attn_classes_extra,
  )

  # NOTE(DefTruth): Allow user to pass in a custom mesh outside this function.
  # We only create a new mesh when `config._mesh` is None. So, that we can
  # support both custom mesh (e.g, hybrid parallelism) and auto-created mesh.
  if config._mesh is None:
    mesh = torch.distributed.device_mesh.init_device_mesh(
      device_type=device_type,
      mesh_shape=config.mesh_shape,
      mesh_dim_names=config.mesh_dim_names,
    )
    config.setup(rank, world_size, device, mesh=mesh)
  elif config._rank is None or config._world_size is None or config._device is None:
    config.setup(rank, world_size, device, mesh=config._mesh)

  # `_cp_config` is the primary cache-dit runtime attribute. `_parallel_config`
  # intentionally aliases the same object for diffusers-style call sites that
  # dereference `parallel_config.context_parallel_config`; cache-dit's
  # `_ContextParallelConfig.context_parallel_config` returns `self` to preserve
  # that access pattern without an extra wrapper object.
  model._cp_config = config  # type: ignore[attr-defined]
  model._parallel_config = config  # type: ignore[attr-defined]

  attention_classes = get_diffusers_attention_classes(attn_classes_extra=attn_classes_extra)

  for module in model.modules():
    if not isinstance(module, attention_classes):
      continue

    processor = getattr(module, "processor", None)
    if processor is None:
      continue
    if not hasattr(processor, "_cp_config"):
      processor._cp_config = None
    if not hasattr(processor, "_parallel_config"):
      processor._parallel_config = None
    processor._cp_config = config
    processor._parallel_config = config

  if cp_plan is None and model._cp_plan is None:
    raise ValueError(
      "`cp_plan` must be provided either as an argument or set in the model's `_cp_plan` attribute."
    )
  cp_plan = cp_plan if cp_plan is not None else model._cp_plan
  # NOTE: _apply_context_parallel only requires the model is an instance of
  # torch.nn.Module, thus, it open the possibility to support non-diffusers
  # models in the future.
  _apply_context_parallel(model, config, cp_plan)
