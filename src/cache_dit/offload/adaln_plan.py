"""Exact AdaLN plan memoization for the MiniMax-H3 family.

Every block of a MiniMax-H3 transformer keeps one `adaln_proj` projection of shape
`(6 * hidden * 3, time_embed_dim)`: 520 MiB per block, 24.2 GiB over 50 blocks, i.e. 40 % of the
62 GiB checkpoint. Its output — the six `shift / scale / gate` tables the block indexes by
`(timestep, modality)` — depends on nothing but the timestep embedding, and the released checkpoints
sample on a **fixed** sigma grid. So the tables are a pure function of a handful of vectors:

* six sampling steps (one per output head), each with its own timestep;
* for reference-image editing, one extra level (`t = 0.999`) that the reference rows are pinned at,
  so those steps carry a two-row timestep vector instead of a one-row one.

This module computes those tables once with the weights in place, keeps the small results
(~19 MiB in total) on the device and leaves the 24.2 GiB of weights on the host. That removes the
per-step host-to-device streaming `--adaln-host-residency` performs, which is the dominant cost of a
bfloat16 run: 24.2 GiB per step at ~29 GB/s is about a second per step, and context parallelism
cannot divide it (every rank reads all 50 tables).

The memoization is **exact**, not an approximation: a table is reused only for a `temb` whose bytes
match the one it was computed from, and the same `linear` weights. A timestep the grid did not
announce is a miss: the weight is moved to the device for that call, the table is recomputed and
stored, and the weight goes back to the host — correct, just as slow as the unprecomputed path.

Reference implementations: sglang's `minimax_h3_adaln_online` (config
`_configure_online_adaln`, "avoid keeping 24.2 GiB of constant AdaLN projections resident") and
vllm-omni's `MiniMaxH3RuntimeAdalnCache` ("exact AdaLN memoization ... reused only for identical
inputs and unchanged weights").
"""

from typing import Dict, List, Sequence, Tuple

import torch
from diffusers.models.modeling_utils import get_parameter_dtype
from torch.distributed.tensor import DTensor

from ..logger import init_logger

logger = init_logger(__name__)

# Report the hit rate once the cache is clearly not being used as intended, and stop growing a table
# that a mistaken grid keeps missing on.
_MISS_REPORT_EVERY = 64
_MISS_CACHE_LIMIT = 128


def _temb_key(temb: torch.Tensor) -> bytes:
  """Return a byte-exact identity of one timestep-embedding batch.

  The tables are keyed on the embedding rather than on the timestep values: two grids that produce
  the same embedding must share the table, and a value that differs in the last bit must not.

  :param temb: Timestep embedding of shape `(num_timesteps, time_embed_dim)`.
  :returns: The raw bytes of the embedding, in float32.
  """
  return temb.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()


class _TembKeyMemo:
  """Cache the byte key of the timestep embedding shared by every block of one step.

  `forward` builds one `temb` per sampling step and hands the same tensor object to all 50 blocks,
  so the identity check keeps the conversion (a device-to-host copy) at once per step instead of
  once per block.
  """

  def __init__(self):
    self._temb: torch.Tensor | None = None
    self._key: bytes | None = None

  def __call__(self, temb: torch.Tensor) -> bytes:
    if temb is not self._temb:
      self._temb = temb
      self._key = _temb_key(temb)
    return self._key


def _modulation_modules(transformer: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
  """Collect every module that projects the timestep embedding into block modulation.

  :param transformer: MiniMax-H3(-Image) transformer.
  :returns: `(name, module)` pairs holding an `adaln_proj` projection.
  """
  return [(name, module) for name, module in transformer.named_modules()
          if name.endswith("adaln_proj") and hasattr(module, "linear")]


def adaln_plan_tembeddings(
  transformer: torch.nn.Module,
  timestep_sets: Sequence[Sequence[float]],
) -> List[torch.Tensor]:
  """Build the timestep embeddings a fixed sampling grid will produce.

  :param transformer: MiniMax-H3(-Image) transformer whose `time_proj` / `time_embedder` define the
    embedding.
  :param timestep_sets: One sequence of timestep values per distinct embedding, already in the order
    the pipeline's `torch.unique(..., sorted=True)` produces.
  :returns: One embedding per entry of `timestep_sets`, bit-identical to what `forward` builds.
  """
  embedder_dtype = get_parameter_dtype(transformer.time_embedder)
  device = next(transformer.time_embedder.parameters()).device
  embeddings = []
  for values in timestep_sets:
    timesteps = torch.tensor(list(values), dtype=torch.float32, device=device)
    embeddings.append(transformer.time_embedder(
      transformer.time_proj(timesteps).to(embedder_dtype)))
  return embeddings


def install_adaln_plan_cache(
  transformer: torch.nn.Module,
  tembeddings: Sequence[torch.Tensor],
  compute_device: str | torch.device,
  keep_weights_on: str | torch.device = "cpu",
) -> Tuple[int, int]:
  """Precompute the AdaLN tables of every block and move their weights off the device.

  The tables are built one block at a time: 520 MiB of projection is onloaded, used, and parked
  again, so the device never holds more than one of them. Building them all up front would need the
  full 62 GiB of checkpoint resident even though 24 GiB of it is dead weight afterwards.

  :param transformer: MiniMax-H3(-Image) transformer, already on its execution device and already
    parallelized (a tensor-parallel plan shards `adaln_proj.linear`, and the tables must be built
    from the sharded projection so that they carry the all-gathered result).
  :param tembeddings: Timestep embeddings from :func:`adaln_plan_tembeddings`.
  :param compute_device: Device the forward runs on.
  :param keep_weights_on: Device the `adaln_proj` weights are parked on afterwards.
  :returns: `(number of plans, number of modulation modules)`.
  """
  modules = _modulation_modules(transformer)
  if not modules:
    raise ValueError("No `adaln_proj` modulation table was found on this transformer; "
                     "--adaln-online only applies to the MiniMax-H3 family.")
  key_memo = _TembKeyMemo()
  plans: Dict[int, Dict[bytes, Tuple[torch.Tensor, ...]]] = {
    id(module): {}
    for _, module in modules
  }
  stats = {"hits": 0, "misses": 0}
  parked = 0
  for _, module in modules:
    if module.linear.weight.device != torch.device(compute_device):
      module.to(compute_device)
    for temb in tembeddings:
      plans[id(module)][key_memo(temb)] = module(temb)
    # A tensor-parallel plan shards the table, and a DTensor cannot be moved to the host: under TP
    # the shard stays resident (it is already divided by the TP size) and only the plans are reused.
    if not isinstance(module.linear.weight, DTensor):
      module.to(keep_weights_on)
      parked += 1

  device = torch.device(compute_device)

  def _install(module: torch.nn.Module) -> None:
    module_plans = plans[id(module)]
    original_forward = module.forward
    weight_device = module.linear.weight.device

    def new_forward(temb, *args, **kwargs):
      key = key_memo(temb)
      plan = module_plans.get(key)
      if plan is not None:
        stats["hits"] += 1
        return plan
      stats["misses"] += 1
      if stats["misses"] == 1:
        logger.warning(
          "AdaLN plan cache miss: a timestep outside the announced sampling grid reached %s. The "
          "table is recomputed with the weights on %s and cached.", module.__class__.__name__,
          weight_device)
      elif stats["misses"] == _MISS_REPORT_EVERY:
        logger.warning(
          "AdaLN plan cache: %d misses against %d hits; the announced sampling grid does not match "
          "the one the pipeline runs, so --adaln-online is not saving anything.", stats["misses"],
          stats["hits"])
      if module.linear.weight.device != device:
        module.to(device)
      plan = original_forward(temb, *args, **kwargs)
      # Cap the table: a grid the caller did not announce would otherwise grow it without bound.
      if stats["misses"] <= _MISS_CACHE_LIMIT:
        module_plans[key] = plan
      if module.linear.weight.device != weight_device:
        module.to(weight_device)
      return plan

    module.forward = new_forward

  for _, module in modules:
    _install(module)

  logger.info(
    "Installed %d AdaLN plans per block over %d modulation tables; %d tables parked on %s "
    "(the rest stay resident).", len(tembeddings), len(modules), parked, keep_weights_on)
  return len(tembeddings) * len(modules), len(modules)


def timestep_sets_from_pipeline(pipe) -> List[List[float]]:
  """Read the fixed sampling grid of a MiniMax-H3 pipeline.

  :param pipe: MiniMax-H3(-Image) modular pipeline.
  :returns: One sorted timestep vector per distinct embedding the grid produces.
  :raises ValueError: When the pipeline does not expose its `sample_sigmas`.
  """
  sigmas = getattr(getattr(pipe, "config", None), "sample_sigmas", None)
  if sigmas is None:
    raise ValueError("--adaln-online reads the checkpoint's fixed sampling grid from "
                     "`config.sample_sigmas`, which this pipeline does not expose.")
  return minimax_h3_adaln_timestep_sets(sigmas, getattr(pipe, "reference_noise_aug", None))


def minimax_h3_adaln_timestep_sets(sigmas: Sequence[float],
                                   reference_level: float | None = None) -> List[List[float]]:
  """Turn a MiniMax-H3 sampling grid into the distinct timestep vectors it visits.

  The schedule is `timesteps = 1 - sigmas[:-1]`; the reference rows of an edit request are pinned at
  `reference_level`, so those steps carry a two-row vector sorted ascending, exactly as
  `torch.unique(..., sorted=True)` orders them.

  :param sigmas: The checkpoint's `sample_sigmas`, including the trailing `0.0`.
  :param reference_level: Conditioning level the reference rows are held at, or None for a
    text-to-image-only run.
  :returns: One sorted timestep vector per distinct embedding.
  """
  timesteps = [1.0 - float(sigma) for sigma in sigmas[:-1]]
  sets = [[step] for step in timesteps]
  if reference_level is not None:
    level = float(reference_level)
    for step in timesteps:
      if step != level:
        sets.append(sorted([step, level]))
  return sets
