"""MAGI-2 Preview distributed parallelism planners.

Both planners are registered on ``Magi2TransformerBlock`` because that is the module the block
adapter points at (the pre/post adapters stay outside the cached region), and cache-dit
parallelizes the adapter's transformer. The model owns both parallel views, so neither planner
builds a DTensor plan or a cache-dit hook plan:

* **Context parallelism**: `Magi2PreviewTransformer.forward` shards the packed token sequence itself
  (`Magi2SequenceDispatcher`) and undoes the split on the way out; its packed attention performs the
  Ulysses head/sequence exchange with the released uneven `split_sizes`. Every sequence tensor is
  therefore already split at the block boundary, so the plan is empty and the planner only has to
  install the process group.
* **Tensor parallelism**: the released checkpoint stores flattened grouped-linear weights and a
  flattened `(head, expert)` MoE layout, and the block shards itself (`shard_for_tp`). A DTensor
  `parallelize_module` plan cannot express either layout (`Magi2GroupedLinear` is not an `nn.Linear`,
  and the MoE shards a head axis, never the expert axis).

Ring attention is refused (the released kernels implement Ulysses only). The hybrid CP+TP layout is
refused as well: the released attention exchange and the Ulysses token split must share one axis, and
the 2-GPU hybrid degenerates to plain TP anyway.
"""

import torch.distributed as dist

from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

__all__ = [
  "Magi2PreviewContextParallelismPlanner",
  "Magi2PreviewTensorParallelismPlanner",
]


def _install_ulysses_group(parallelism_config: ParallelismConfig) -> None:
  """Install the token-sharding group the model's own dispatcher reads.

  :param parallelism_config: Requested parallelism configuration.
  :raises NotImplementedError: When the requested layout is a CP+TP hybrid.
  """

  from ...models.pipelines.magi2.parallel import set_magi2_groups

  if int(getattr(parallelism_config, "tp_size", 1) or 1) > 1:
    raise NotImplementedError(
      "MAGI-2 does not support a CP+TP hybrid layout: the packed attention exchange and the token "
      "split must share one axis, and `Magi2GroupedLinear` shards with the same group the MoE head "
      "axis uses. Run `--parallel ulysses` or `--parallel tp` instead.")
  if not dist.is_available() or not dist.is_initialized():
    # Single-process bring-up (tests, `--parallel` not requested): nothing to shard.
    return
  world = dist.get_world_size()
  ulysses_size = int(getattr(parallelism_config, "ulysses_size", 1) or 1)
  if ulysses_size != world:
    raise NotImplementedError(
      f"MAGI-2 context parallelism expects the Ulysses group to cover every worker; got "
      f"ulysses_size={ulysses_size} with world_size={world}.")
  set_magi2_groups(ulysses_group=dist.group.WORLD)


@ContextParallelismPlannerRegister.register("Magi2TransformerBlock")
class Magi2PreviewContextParallelismPlanner(ContextParallelismPlanner):
  """Context-parallel planner for MAGI-2 Preview (model-owned Ulysses).

  Returns an empty hook plan; the model shards and gathers the packed sequence inside its own forward
  and the packed attention carries the uneven per-rank `split_sizes`.
  """

  def _apply(
    self,
    transformer=None,
    parallelism_config: ParallelismConfig = None,
    **kwargs,
  ) -> dict:
    """Validate the requested CP topology and install the group.

    :param transformer: MAGI-2 layer-stack container.
    :param parallelism_config: Requested parallelism configuration.
    :param kwargs: Unused planner options.
    :returns: An empty CP plan; the model splits the sequence itself.
    :raises NotImplementedError: For Ring attention, or a CP+TP hybrid.
    """

    name = type(transformer).__name__ if transformer is not None else "Magi2TransformerBlock"
    ring_size = int(getattr(parallelism_config, "ring_size", 1) or 1) if parallelism_config else 1
    if ring_size > 1:
      raise NotImplementedError(
        f"Context parallelism for MAGI-2 Preview ({name}) supports Ulysses only; got ring_size={ring_size}. "
        "The released packed attention exchanges heads and sequence with a fused Ulysses all-to-all "
        "that carries uneven per-rank `split_sizes`; Ring attention would need a different kernel and a "
        "different RoPE/position bookkeeping. Use `--parallel ulysses`.")
    _install_ulysses_group(parallelism_config)
    return {}


@TensorParallelismPlannerRegister.register("Magi2TransformerBlock")
class Magi2PreviewTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor-parallel planner for MAGI-2 Preview (model-owned sharding).

  The released checkpoint is only distributed through the module's own layout: grouped linear shards
  its column/row axis with a checkpoint-aware transform, and the multi-head MoE shards the MoE head
  axis in contiguous `(head, expert)` row blocks. Both are expressed by `shard_for_tp`, so no DTensor
  `parallelize_module` plan is returned.
  """

  def _apply(
    self,
    transformer=None,
    parallelism_config: ParallelismConfig = None,
    **kwargs,
  ):
    """Install the TP group and shard the layer stack.

    :param transformer: MAGI-2 layer-stack container.
    :param parallelism_config: Requested parallelism configuration.
    :param kwargs: Unused planner options.
    :returns: ``(transformer, [])`` - cache-dit records no DTensor layer plans for this model.
    :raises NotImplementedError: When the model classes are not the vendored MAGI-2 ones.
    """

    from ...models.pipelines.magi2.transformer import Magi2TransformerBlock

    if not isinstance(transformer, Magi2TransformerBlock):
      raise NotImplementedError(
        f"MAGI-2 tensor parallelism needs the vendored `Magi2TransformerBlock`, got "
        f"{type(transformer).__name__}. Its grouped-linear / MoE-head layout cannot be expressed as a "
        "DTensor `parallelize_module` plan.")

    from ...models.pipelines.magi2.parallel import get_magi2_tp_group, set_magi2_groups

    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    set_magi2_groups(tp_group=tp_mesh.get_group())
    transformer.shard_for_tp(get_magi2_tp_group())
    return transformer, []
