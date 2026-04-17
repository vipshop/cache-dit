"""Adapted from diffusers' context-parallel hook runtime with cache-dit extensions."""

import copy
import functools
import inspect
from dataclasses import dataclass
from typing import Type

import torch
import torch.distributed as dist

if torch.distributed.is_available():
  import torch.distributed._functional_collectives as funcol

from ...logger import init_logger
from ._distributed_primitives import _gather_size
from ._hooks import HookRegistry, ModelHook, unwrap_module
from ._modeling_parallel import (
  _ContextParallelConfig,
  _ContextParallelInput,
  _ContextParallelOutput,
)

logger = init_logger(__name__)

_CONTEXT_PARALLEL_INPUT_HOOK_TEMPLATE = "cp_input---{}"
_CONTEXT_PARALLEL_OUTPUT_HOOK_TEMPLATE = "cp_output---{}"


@dataclass
class _ModuleForwardMetadata:
  """Cache forward signature parameter indices for one module class."""

  cached_parameter_indices: dict[str, int] | None = None
  _cls: Type | None = None

  def get_parameter_from_args_kwargs(self, identifier: str, args=(), kwargs=None):
    """Resolve one forward parameter from positional and keyword args."""

    kwargs = kwargs or {}

    if identifier in kwargs:
      return kwargs[identifier], True, None

    if self.cached_parameter_indices is not None:
      index = self.cached_parameter_indices.get(identifier)
      if index is None:
        raise ValueError(f"Parameter '{identifier}' not found in cached indices.")
      return args[index], False, index

    if self._cls is None:
      raise ValueError("Model class is not set for metadata.")

    parameters = list(inspect.signature(self._cls.forward).parameters.keys())[1:]
    self.cached_parameter_indices = {param: index for index, param in enumerate(parameters)}

    if identifier not in self.cached_parameter_indices:
      raise ValueError(
        f"Parameter '{identifier}' not found in function signature but was requested.")

    index = self.cached_parameter_indices[identifier]
    if index >= len(args):
      raise ValueError(f"Expected {index} arguments but got {len(args)}.")

    return args[index], False, index


def _apply_context_parallel(
  module: torch.nn.Module,
  parallel_config: _ContextParallelConfig,
  plan: dict,
) -> None:
  """Attach split/gather hooks according to one CP plan.

  :param module: Root module to patch.
  :param parallel_config: Runtime CP configuration.
  :param plan: Module-level CP plan.
  """

  logger.debug(f"Applying context parallel with mesh {parallel_config._mesh} and plan: {plan}")

  for module_id, cp_model_plan in plan.items():
    submodule = _get_submodule_by_name(module, module_id)
    submodules = submodule if isinstance(submodule, list) else [submodule]

    for current_module in submodules:
      registry = HookRegistry.check_if_exists_or_initialize(current_module)
      # A module entry may expand to multiple hooks when the plan uses subplans.
      # The most common case is one split hook plus one gather hook on the same
      # module key, e.g. `[input_plan, output_plan]` for `attn.to_v`.
      for subplan_index, current_plan in _iter_cp_module_plans(cp_model_plan):
        hook, hook_name = _build_cp_hook(current_plan, parallel_config, module_id, subplan_index)
        registry.register_hook(hook, hook_name)


def _remove_context_parallel(module: torch.nn.Module, plan: dict) -> None:
  """Remove previously attached CP hooks.

  :param module: Root module whose hooks should be removed.
  :param plan: Plan used when hooks were registered.
  """

  for module_id, cp_model_plan in plan.items():
    submodule = _get_submodule_by_name(module, module_id)
    submodules = submodule if isinstance(submodule, list) else [submodule]

    for current_module in submodules:
      registry = HookRegistry.check_if_exists_or_initialize(current_module)
      for subplan_index, current_plan in _iter_cp_module_plans(cp_model_plan):
        _, hook_name = _build_cp_hook(current_plan, None, module_id, subplan_index)
        registry.remove_hook(hook_name)


def _iter_cp_module_plans(cp_model_plan):
  """Yield one or more hook plans for a module entry.

  A plain dict or plain output plan produces one `(None, plan)` pair.

  A list/tuple subplan container registers multiple hooks on the same module in the
  listed order. Example:

    "attn.to_v": [
      {"input": _ContextParallelInput(split_dim=1, expected_dims=3)},
      _ContextParallelOutput(gather_dim=1, expected_dims=3),
    ]

  The runtime turns this into `cp_input---attn.to_v---0` and
  `cp_output---attn.to_v---1`. Because later hooks wrap earlier hooks in
  `HookRegistry`, `[input_plan, output_plan]` yields split in `pre_forward` and
  gather in `post_forward`, which matches the intuitive "split input, gather
  output" semantics.
  """

  if _is_subplan_container(cp_model_plan):
    return list(enumerate(cp_model_plan))
  return [(None, cp_model_plan)]


def _build_cp_hook(cp_model_plan, parallel_config, module_id: str, subplan_index: int | None):
  """Build one split/gather hook plus its stable registry name.

  Note that an output tuple like
  `(_ContextParallelOutput(...), _ContextParallelOutput(...))` is still one output
  plan for a multi-tensor module return, not two independent subplans.
  """

  if isinstance(cp_model_plan, dict):
    hook = _ContextParallelSplitHook(cp_model_plan,
                                     parallel_config) if parallel_config is not None else None
    hook_name = _CONTEXT_PARALLEL_INPUT_HOOK_TEMPLATE.format(module_id)
  elif _is_output_plan(cp_model_plan):
    output_plan = [cp_model_plan] if _is_cp_output_like(cp_model_plan) else list(cp_model_plan)
    hook = _ContextParallelGatherHook(output_plan,
                                      parallel_config) if parallel_config is not None else None
    hook_name = _CONTEXT_PARALLEL_OUTPUT_HOOK_TEMPLATE.format(module_id)
  else:
    raise ValueError(f"Unsupported context parallel model plan type: {type(cp_model_plan)}")

  if subplan_index is not None:
    hook_name = f"{hook_name}---{subplan_index}"
  return hook, hook_name


class _ContextParallelSplitHook(ModelHook):
  """Shard configured inputs before forward and optionally shard outputs after forward."""

  def __init__(self, metadata: dict, parallel_config: _ContextParallelConfig) -> None:
    super().__init__()
    self.metadata = metadata
    self.parallel_config = parallel_config
    self.module_forward_metadata: _ModuleForwardMetadata | None = None

  def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
    cls = unwrap_module(module).__class__
    self.module_forward_metadata = _ModuleForwardMetadata(_cls=cls)
    return module

  def pre_forward(self, module: torch.nn.Module, *args, **kwargs):
    args_list = list(args)

    for name, cpm in self.metadata.items():
      if _is_cp_input_like(cpm) and getattr(cpm, "split_output", False):
        continue

      input_val, is_kwarg, index = self.module_forward_metadata.get_parameter_from_args_kwargs(
        name, args_list, kwargs)

      if input_val is None:
        continue

      if isinstance(input_val, torch.Tensor):
        input_val = self._prepare_cp_input(input_val, cpm)
      elif isinstance(input_val, (list, tuple)):
        if len(input_val) != len(cpm):
          raise ValueError(
            f"Expected input model plan to have {len(input_val)} elements, but got {len(cpm)}.")
        sharded_input_val = []
        for current_input, current_plan in zip(input_val, cpm):
          if torch.is_tensor(current_input) and not getattr(current_plan, "split_output", False):
            current_input = self._prepare_cp_input(current_input, current_plan)
          sharded_input_val.append(current_input)
        input_val = sharded_input_val
      else:
        raise ValueError(f"Unsupported input type: {type(input_val)}")

      if is_kwarg:
        kwargs[name] = input_val
      elif index is not None and index < len(args_list):
        args_list[index] = input_val
      else:
        raise ValueError(f"Unexpected error occurred while processing input '{name}'.")

    return tuple(args_list), kwargs

  def post_forward(self, module: torch.nn.Module, output):
    is_tensor = isinstance(output, torch.Tensor)
    is_tensor_list = isinstance(output,
                                (list, tuple)) and all(isinstance(x, torch.Tensor) for x in output)

    if not is_tensor and not is_tensor_list:
      raise ValueError(
        f"Expected output to be a tensor or a list/tuple of tensors, but got {type(output)}.")

    output_list = [output] if is_tensor else list(output)
    for index, cpm in self.metadata.items():
      if not _is_cp_input_like(cpm) or not getattr(cpm, "split_output", False):
        continue
      if index >= len(output_list):
        raise ValueError(f"Index {index} out of bounds for output of length {len(output_list)}.")
      output_list[index] = self._prepare_cp_input(output_list[index], cpm)

    return output_list[0] if is_tensor else tuple(output_list)

  def _prepare_cp_input(self, tensor: torch.Tensor, cp_input) -> torch.Tensor:
    expected_dims = getattr(cp_input, "expected_dims", None)
    if expected_dims is not None and tensor.dim() != expected_dims:
      logger.warning_once(
        "Expected input tensor to have %s dimensions, but got %s dimensions; split will be skipped.",
        expected_dims,
        tensor.dim(),
      )
      return tensor

    split_dim = getattr(cp_input, "split_dim")
    if self.parallel_config.ulysses_anything:
      return _PartitionAnythingSharder.shard_anything(tensor, split_dim,
                                                      self.parallel_config._flattened_mesh)
    return _EquipartitionSharder.shard(tensor, split_dim, self.parallel_config._flattened_mesh)


class _ContextParallelGatherHook(ModelHook):
  """Gather configured outputs after forward."""

  def __init__(self, metadata: list, parallel_config: _ContextParallelConfig) -> None:
    super().__init__()
    self.metadata = metadata
    self.parallel_config = parallel_config

  def post_forward(self, module: torch.nn.Module, output):
    is_tensor = isinstance(output, torch.Tensor)

    if is_tensor:
      output_list = [output]
    elif isinstance(output, (list, tuple)) and all(isinstance(x, torch.Tensor) for x in output):
      output_list = list(output)
    else:
      raise ValueError(
        f"Expected output to be a tensor or a list/tuple of tensors, but got {type(output)}.")

    if len(output_list) != len(self.metadata):
      raise ValueError(
        f"Expected output to have {len(self.metadata)} elements, but got {len(output_list)}.")

    for index, cpm in enumerate(self.metadata):
      if cpm is None:
        continue
      gather_dim = getattr(cpm, "gather_dim")
      if self.parallel_config.ulysses_anything:
        output_list[index] = _PartitionAnythingSharder.unshard_anything(
          output_list[index], gather_dim, self.parallel_config._flattened_mesh)
      else:
        output_list[index] = _EquipartitionSharder.unshard(output_list[index], gather_dim,
                                                           self.parallel_config._flattened_mesh)

    return output_list[0] if is_tensor else tuple(output_list)


class _AllGatherFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, tensor, dim, group):
    ctx.dim = dim
    ctx.group = group
    ctx.world_size = dist.get_world_size(group)
    ctx.rank = dist.get_rank(group)
    return funcol.all_gather_tensor(tensor, dim, group=group)

  @staticmethod
  def backward(ctx, grad_output):
    grad_chunks = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)
    return grad_chunks[ctx.rank], None, None


class _EquipartitionSharder:

  @classmethod
  def shard(cls, tensor: torch.Tensor, dim: int,
            mesh: torch.distributed.device_mesh.DeviceMesh) -> torch.Tensor:
    """Shard a tensor evenly across the flattened CP mesh."""

    assert tensor.size()[dim] % mesh.size() == 0, (
      "Tensor size along dimension to be sharded must be divisible by mesh size")
    return tensor.chunk(mesh.size(), dim=dim)[dist.get_rank(mesh.get_group())]

  @classmethod
  def unshard(cls, tensor: torch.Tensor, dim: int,
              mesh: torch.distributed.device_mesh.DeviceMesh) -> torch.Tensor:
    """Gather an evenly sharded tensor back to each rank."""

    return _AllGatherFunction.apply(tensor.contiguous(), dim, mesh.get_group())


class _AllGatherAnythingFunction(torch.autograd.Function):

  @staticmethod
  def forward(ctx, tensor: torch.Tensor, dim: int, group: dist.ProcessGroup):
    ctx.dim = dim
    ctx.group = group
    ctx.world_size = dist.get_world_size(group)
    ctx.rank = dist.get_rank(group)
    return _all_gather_anything(tensor, dim, group)

  @staticmethod
  def backward(ctx, grad_output):
    grad_splits = torch.tensor_split(grad_output, ctx.world_size, dim=ctx.dim)
    return grad_splits[ctx.rank], None, None


class _PartitionAnythingSharder:

  @classmethod
  def shard_anything(
    cls,
    tensor: torch.Tensor,
    dim: int,
    mesh: torch.distributed.device_mesh.DeviceMesh,
  ) -> torch.Tensor:
    """Shard a tensor with possibly uneven partition sizes."""

    assert tensor.size()[dim] >= mesh.size(), (
      f"Cannot shard tensor of size {tensor.size()} along dim {dim} across mesh of size {mesh.size()}."
    )
    return tensor.tensor_split(mesh.size(), dim=dim)[dist.get_rank(mesh.get_group())]

  @classmethod
  def unshard_anything(
    cls,
    tensor: torch.Tensor,
    dim: int,
    mesh: torch.distributed.device_mesh.DeviceMesh,
  ) -> torch.Tensor:
    """Gather an unevenly sharded tensor back to each rank."""

    return _AllGatherAnythingFunction.apply(tensor.contiguous(), dim, mesh.get_group())


@functools.lru_cache(maxsize=64)
def _fill_gather_shapes(shape: tuple[int, ...], gather_dims: tuple[int, ...],
                        dim: int) -> list[list[int]]:
  gather_shapes = []
  for gather_dim in gather_dims:
    rank_shape = list(copy.deepcopy(shape))
    rank_shape[dim] = gather_dim
    gather_shapes.append(rank_shape)
  return gather_shapes


def _all_gather_anything(tensor: torch.Tensor, dim: int, group: dist.ProcessGroup) -> torch.Tensor:
  tensor = tensor.contiguous()
  shape = tensor.shape
  gather_dims = tuple(_gather_size(shape[dim], group))
  gather_shapes = _fill_gather_shapes(tuple(shape), gather_dims, dim)
  gathered_tensors = [
    torch.empty(shape, device=tensor.device, dtype=tensor.dtype) for shape in gather_shapes
  ]
  dist.all_gather(gathered_tensors, tensor, group=group)
  return torch.cat(gathered_tensors, dim=dim)


def _get_submodule_by_name(model: torch.nn.Module,
                           name: str) -> torch.nn.Module | list[torch.nn.Module]:
  if name.count("*") > 1:
    raise ValueError("Wildcard '*' can only be used once in the name")
  return _find_submodule_by_name(model, name)


def _find_submodule_by_name(model: torch.nn.Module,
                            name: str) -> torch.nn.Module | list[torch.nn.Module]:
  if name == "":
    return model
  first_atom, remaining_name = name.split(".", 1) if "." in name else (name, "")
  if first_atom == "*":
    if not isinstance(model, torch.nn.ModuleList):
      raise ValueError("Wildcard '*' can only be used with ModuleList")
    submodules = []
    for submodule in model:
      subsubmodules = _find_submodule_by_name(submodule, remaining_name)
      if not isinstance(subsubmodules, list):
        subsubmodules = list(subsubmodules.values()) if isinstance(
          subsubmodules, torch.nn.ModuleDict) else [subsubmodules]
      submodules.extend(subsubmodules)
    return submodules

  if not hasattr(model, first_atom):
    raise ValueError(f"'{first_atom}' is not a submodule of '{model.__class__.__name__}'")

  submodule = getattr(model, first_atom)
  if isinstance(submodule, torch.nn.ModuleDict):
    if remaining_name:
      raise ValueError(
        f"Cannot access submodule '{remaining_name}' of ModuleDict '{first_atom}' directly. Please specify the key first."
      )
    values = list(submodule.values())
    if not all(isinstance(value, torch.nn.Module) for value in values):
      raise ValueError(f"ModuleDict '{first_atom}' contains a non-module value.")
    return values
  return _find_submodule_by_name(submodule, remaining_name)


def _is_cp_input_like(value) -> bool:
  return isinstance(value, _ContextParallelInput) or (hasattr(value, "split_dim")
                                                      and hasattr(value, "split_output"))


def _is_cp_output_like(value) -> bool:
  return isinstance(value, _ContextParallelOutput) or hasattr(value, "gather_dim")


def _is_output_plan(value) -> bool:
  if _is_cp_output_like(value):
    return True
  if isinstance(value, (list, tuple)):
    return all(item is None or _is_cp_output_like(item) for item in value)
  return False


def _is_subplan_container(value) -> bool:
  """Return whether a list/tuple encodes multiple hooks for one module key.

  We deliberately exclude pure output tuples here. For example,
  `(_ContextParallelOutput(...), _ContextParallelOutput(...))` means one gather hook
  over a tuple-valued module output, while `[dict_input_plan, output_plan]` means two
  hooks attached to the same module.
  """

  if not isinstance(value, (list, tuple)) or len(value) == 0:
    return False
  if _is_output_plan(value):
    return False
  return all(isinstance(item, dict) or _is_output_plan(item) for item in value)
