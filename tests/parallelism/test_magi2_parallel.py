"""CPU tests for the MAGI-2 Preview CP/TP planners and the in-place TP sharding.

The parallel planners are registered under the transformer class name, so these tests also guard the
registry wiring (`No planner registered under name: Magi2PreviewTransformer` would otherwise only
show up on a 2-GPU run). All checks are pure CPU tensor slicing: no process group is created.
"""
import pytest
import torch

from cache_dit.distributed.transformers.planners import (
  _activate_cp_planners,
  _activate_tp_planners,
)
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.models.pipelines.magi2.configuration import (
  Magi2MHCConfig,
  Magi2MoEConfig,
  Magi2PreviewConfig,
)
from cache_dit.models.pipelines.magi2.parallel import (
  Magi2ParallelGroup,
  get_magi2_tp_group,
  reset_magi2_groups,
  set_magi2_groups,
)
from cache_dit.models.pipelines.magi2.transformer import Magi2MultiHeadMoE, Magi2PreviewTransformer

NUM_LAYERS = 2


class _FakeConfig:
  """Minimal `ParallelismConfig` surface the planners read."""

  def __init__(self, ring_size: int = 1, ulysses_size: int = 2) -> None:
    self.ring_size = ring_size
    self.ulysses_size = ulysses_size


@pytest.fixture(autouse=True)
def _reset_groups():
  reset_magi2_groups()
  yield
  reset_magi2_groups()


def tiny_config() -> Magi2PreviewConfig:
  """Tiny validated config covering a multimodal dense layer and one MoE layer."""

  config = Magi2PreviewConfig(
    num_layers=NUM_LAYERS,
    # 192 keeps the released dense-MLP rounding (int(192 * 4 * 2 / 3) // 128 * 128 = 512) and divides
    # by the 2-rank TP group on both the attention heads (24) and the FFN width.
    hidden_size=192,
    head_dim=8,
    num_query_groups=24,
    video_in_channels=4,
    audio_in_channels=4,
    text_in_channels=8,
    multimodal_layers=(0, ),
    params_dtype=torch.float32,
    mhc=Magi2MHCConfig(num_streams=2, sinkhorn_iterations=2),
    moe=Magi2MoEConfig(
      num_heads=2,
      num_experts=4,
      top_k=2,
      expert_intermediate_size=8,
      shared_expert_intermediate_size=8,
      modality_shared_expert_intermediate_size=8,
      layers=(1, ),
    ),
  )
  config.validate()
  return config


def tiny_transformer(seed: int = 0) -> Magi2PreviewTransformer:
  """Tiny transformer with every parameter filled deterministically."""

  generator = torch.Generator().manual_seed(seed)
  model = Magi2PreviewTransformer(tiny_config()).eval()
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
  return model


def _two_rank_group(rank: int) -> Magi2ParallelGroup:
  return Magi2ParallelGroup(group=None, world_size=2, rank=rank, replicated_sequence=True)


def _moe_modules(model: Magi2PreviewTransformer):
  return [module for module in model.modules() if isinstance(module, Magi2MultiHeadMoE)]


def test_planners_are_registered_under_the_transformer_class():
  """Both planners must resolve from the vendored transformer class name."""

  _activate_cp_planners()
  _activate_tp_planners()
  transformer = tiny_transformer().block
  assert (ContextParallelismPlannerRegister.get_planner(transformer).__name__ ==
          "Magi2PreviewContextParallelismPlanner")
  assert (TensorParallelismPlannerRegister.get_planner(transformer).__name__ ==
          "Magi2PreviewTensorParallelismPlanner")


def test_cp_planner_returns_an_empty_plan():
  """MAGI-2 splits the packed sequence itself, so no cache-dit hook may be installed."""

  _activate_cp_planners()
  transformer = tiny_transformer().block
  planner = ContextParallelismPlannerRegister.get_planner(transformer)()
  plan = planner.apply(transformer=transformer, parallelism_config=_FakeConfig())
  assert plan == {}, f"unexpected CP hook plan: {plan}"


def test_cp_planner_refuses_ring():
  """Ring attention has no released MAGI-2 kernel and must be refused with a reason."""

  _activate_cp_planners()
  transformer = tiny_transformer().block
  planner = ContextParallelismPlannerRegister.get_planner(transformer)()
  with pytest.raises(NotImplementedError, match="Ulysses only"):
    planner.apply(transformer=transformer, parallelism_config=_FakeConfig(ring_size=2))


def _sharded_model(rank: int) -> Magi2PreviewTransformer:
  """Build a transformer whose modules are constructed with a 2-rank TP group."""

  set_magi2_groups(tp_group=_two_rank_group(rank))
  try:
    model = Magi2PreviewTransformer(tiny_config()).eval()
  finally:
    reset_magi2_groups()
  return model


def test_shard_for_tp_matches_loading_a_two_rank_checkpoint():
  """In-place TP sharding must equal loading the same weights into a 2-rank module."""

  full_model = tiny_transformer()
  snapshot = {name: tensor.detach().clone() for name, tensor in full_model.state_dict().items()}

  for rank in (0, 1):
    loaded = _sharded_model(rank)
    loaded.load_weights(list(snapshot.items()))

    sharded = tiny_transformer()
    sharded.shard_for_tp(_two_rank_group(rank))

    for name, tensor in sharded.state_dict().items():
      if name.endswith("router.expert_bias"):
        # `load_weights` copies the released EMA router bias over the main one at the end of loading;
        # `shard_for_tp` keeps whatever the module already holds. Same sharding, different source.
        continue
      torch.testing.assert_close(tensor,
                                 loaded.state_dict()[name],
                                 atol=0,
                                 rtol=0,
                                 msg=f"rank {rank} {name}")


def test_column_parallel_sharding_matches_manual_slicing():
  """Column-parallel sharding must reproduce the released per-segment layout."""

  full_model = tiny_transformer()
  full = full_model.block.layers[0].attention.linear_qkv.weight.detach().clone()
  splits = full_model.block.layers[0].attention.linear_qkv.qkv_splits
  assert splits is not None

  num_experts = full_model.block.layers[0].attention.linear_qkv.num_experts
  grouped = full.view(num_experts, sum(splits), full.shape[-1])

  for rank in (0, 1):
    linear = _sharded_model(rank).block.layers[0].attention.linear_qkv
    per_expert = []
    for expert in range(num_experts):
      pieces = []
      base = 0
      for size in splits:
        local = size // 2
        start = base + rank * local
        pieces.append(grouped[expert, start:start + local])
        base += size
      per_expert.append(torch.cat(pieces, dim=0))
    expected = torch.cat(per_expert, dim=0)
    torch.testing.assert_close(linear.shard_checkpoint_weight(full), expected, atol=0, rtol=0)


def test_row_parallel_sharding_slices_the_input_axis():
  """Row-parallel sharding must split the input features, not the output rows."""

  full_model = tiny_transformer()
  full = full_model.block.layers[0].attention.linear_proj.weight.detach().clone()
  half = full.shape[1] // 2

  for rank, expected in ((0, full[:, :half]), (1, full[:, half:])):
    linear = _sharded_model(rank).block.layers[0].attention.linear_proj
    torch.testing.assert_close(linear.shard_checkpoint_weight(full), expected, atol=0, rtol=0)


def test_moe_shards_the_head_axis_not_the_expert_axis():
  """The MoE shards whole `(head, expert)` row blocks of the head axis."""

  full_model = tiny_transformer()
  full_gate = _moe_modules(full_model)[0].W_gate.detach().clone()
  num_experts = _moe_modules(full_model)[0].num_experts
  heads = _moe_modules(full_model)[0].config.num_heads

  model = tiny_transformer()
  model.shard_for_tp(_two_rank_group(1))
  sharded_moe = _moe_modules(model)[0]
  expected = full_gate[(heads // 2) * num_experts:heads * num_experts]
  torch.testing.assert_close(sharded_moe.W_gate.detach(), expected, atol=0, rtol=0)
  assert sharded_moe.W_gate.shape[0] == (heads // 2) * num_experts


def test_sinks_are_sharded_with_the_head_axis():
  """The learned sink must keep exactly the heads this rank owns."""

  full_model = tiny_transformer()
  sinks = full_model.block.layers[0].attention.sinks.detach().clone()

  model = tiny_transformer()
  model.shard_for_tp(_two_rank_group(1))
  half = sinks.shape[1] // 2
  torch.testing.assert_close(model.block.layers[0].attention.sinks.detach(),
                             sinks[:, half:],
                             atol=0,
                             rtol=0)


def test_grouped_linear_built_after_group_installation_is_sharded():
  """`set_magi2_groups` must reach the modules constructed afterwards."""

  from cache_dit.models.pipelines.magi2.common_layers import make_grouped_linear

  set_magi2_groups(tp_group=_two_rank_group(1))
  group = get_magi2_tp_group()
  assert group.world_size == 2 and group.rank == 1
  linear = make_grouped_linear(8, 16, parallel_mode="column", dtype=torch.float32)
  assert linear.local_out_features == 8
