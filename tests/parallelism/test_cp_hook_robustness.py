"""Regression tests for the two CP-hook behaviours that modular pipelines rely on.

A diffusers modular pipeline (e.g. FLUX.3) calls the transformer with **keyword arguments only** and
omits workflow-specific inputs (an image-to-image capable transformer has no
`image_cond_hidden_states` in a text-to-image call), and it builds the transformer kwargs by
inspecting `transformer.forward`. Both properties used to break the CP hooks:

1. `_ModuleForwardMetadata.get_parameter_from_args_kwargs` raised `Expected N arguments but got 0`
   for an absent optional input instead of reporting it as `None` (skip the split).
2. `HookRegistry.register_hook` wrapped `forward` as `(*args, **kwargs)`, hiding every required
   parameter from the caller's signature introspection.
"""
import inspect

import torch

from cache_dit.distributed.core._context_parallel import _ModuleForwardMetadata
from cache_dit.distributed.core._hooks import HookRegistry, ModelHook


class _TransformerLike(torch.nn.Module):
  """Minimal stand-in for a transformer whose forward takes optional sequence inputs."""

  def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    img_ids: torch.Tensor,
    image_cond_hidden_states: torch.Tensor | None = None,
    return_dict: bool = True,
  ):
    return hidden_states


class _PassThroughHook(ModelHook):

  def __init__(self):
    super().__init__()

  def pre_forward(self, module, *args, **kwargs):
    return args, kwargs

  def post_forward(self, module, output):
    return output


def test_absent_optional_input_resolves_to_none():
  metadata = _ModuleForwardMetadata(_cls=_TransformerLike)

  hidden = torch.zeros(1, 4, 8)
  encoder = torch.zeros(1, 2, 8)
  ids = torch.zeros(4, 4)

  # a modular pipeline passes every provided input as a keyword argument
  value, is_kwarg, index = metadata.get_parameter_from_args_kwargs(
    "hidden_states",
    (),
    {
      "hidden_states": hidden,
      "encoder_hidden_states": encoder,
      "img_ids": ids
    },
  )
  assert is_kwarg and index is None and value is hidden

  # `image_cond_hidden_states` is omitted by a text-to-image call: nothing to split
  value, is_kwarg, index = metadata.get_parameter_from_args_kwargs(
    "image_cond_hidden_states",
    (),
    {
      "hidden_states": hidden,
      "encoder_hidden_states": encoder,
      "img_ids": ids
    },
  )
  assert value is None and is_kwarg is False and index is None

  # positional callers keep resolving by index
  value, is_kwarg, index = metadata.get_parameter_from_args_kwargs(
    "encoder_hidden_states",
    (hidden, encoder),
    {},
  )
  assert value is encoder and is_kwarg is False and index == 1


def test_flux3_cp_plan_matches_forward_signature():
  """Every planned CP input must exist in `forward`, and the gather point must be a real module."""
  from diffusers import Flux3Transformer2DModel

  from cache_dit.distributed.transformers.flux3 import Flux3ContextParallelismPlanner

  transformer = Flux3Transformer2DModel(
    in_channels=8,
    cond_in_channels=8,
    num_layers=1,
    num_single_layers=1,
    attention_head_dim=8,
    num_attention_heads=2,
    joint_attention_dim=16,
    timestep_channels=8,
    axes_dims_rope=(2, 2, 2, 2),
  )
  plan = Flux3ContextParallelismPlanner()._apply(transformer=transformer, parallelism_config=None)

  parameters = inspect.signature(transformer.forward).parameters
  for name, spec in plan[""].items():
    assert name in parameters, f"planned CP input '{name}' is not a forward parameter"
    # sequence tensors are [B, S, C]; the id tables are [S, 4]
    expected_dims = 3 if name in (
      "hidden_states",
      "encoder_hidden_states",
      "image_cond_hidden_states",
    ) else 2
    assert spec.split_dim == expected_dims - 2
    assert spec.expected_dims == expected_dims

  gather_plan = plan["proj_out"]
  assert getattr(gather_plan, "gather_dim", None) == 1
  assert dict(transformer.named_modules()).get("proj_out") is transformer.proj_out


def test_hook_registry_preserves_forward_signature():
  module = _TransformerLike()
  signature_before = tuple(inspect.signature(module.forward).parameters)

  HookRegistry.check_if_exists_or_initialize(module).register_hook(
    _PassThroughHook(),
    "test_hook",
  )

  assert tuple(inspect.signature(module.forward).parameters) == signature_before
  # a fixed input still works through the hook
  out = module(
    hidden_states=torch.zeros(1, 4, 8),
    encoder_hidden_states=torch.zeros(1, 2, 8),
    img_ids=torch.zeros(4, 4),
  )
  assert out.shape == (1, 4, 8)
