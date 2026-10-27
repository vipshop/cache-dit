"""Kandinsky 6.0 transformer distributed parallelism planners (CP + TP).

`Kandinsky6Transformer3DModel` denoises two streams through one fused block stack: the video stream
(`vis`, 4096 channels) and the audio stream (`aud`, 2048 channels). Each
`Kandinsky6FusedTransformerDecoderBlock` owns six attentions — video self / video->text cross /
audio self / audio->text cross / video-query-audio-KV / audio-query-video-KV — and both streams are
fused by the two cross-modal attentions.

That shape drives both planners:

* CP shards **only the video stream**. The audio stream (a few hundred tokens) stays replicated:
  its self-attention is then computed redundantly on every rank, which is exact, and
  `va_cross_attention` (video queries, audio KV) keeps its KV complete. `av_cross_attention`
  (audio queries, video KV) is the one place that needs the video shard back, so its processor
  all-gathers the video activation and the video rope before the attention.
* TP shards the six attentions and the two FFNs of every fused block. Both streams use the same
  column grid, so an attention whose KV comes from the other stream stays head-aligned.
"""

import functools
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_kandinsky6 import (
  Kandinsky6AttnProcessor,
  Kandinsky6FusedTransformerDecoderBlock,
  Kandinsky6Transformer3DModel,
)
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
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

# Upstream `__call__` captured before the class-level CP patch is installed. Unarmed processors hand
# control back to it, so a cp-free run stays byte-identical to upstream.
_CP_PATCH_ORIGINALS: dict = {}
_ATTN_PROCESSOR_PATCHED = False

# Module attributes the planners tag so the attention patch can tell the six attention roles apart.
_KV_GATHER_ATTR = "_cp_gather_kv"
_SELF_DISTRIBUTED_ATTR = "_cp_self_distributed"


def _cp_is_armed(processor) -> bool:
  """Whether `processor` was armed by the CP runtime with a multi-rank `_cp_config`.

  :param processor: Attention processor instance.
  :returns: True when context parallelism is active on more than one rank.
  """
  cp_config = getattr(processor, "_cp_config", None)
  return cp_config is not None and cp_config._world_size > 1


def _cp_all_gather(x: torch.Tensor, dim: int, processor) -> torch.Tensor:
  """Concatenate one tensor's shards along `dim` across the CP group.

  The CP runtime equipartitions every sequence, so the shards are equally sized and a plain
  `all_gather` into preallocated buffers is enough.

  :param x: Local shard.
  :param dim: Dimension to concatenate on.
  :param processor: Attention processor carrying the active `_cp_config`.
  :returns: The gathered tensor.
  """
  mesh = processor._cp_config._flattened_mesh
  group = mesh.get_group()
  x = x.contiguous()
  chunks = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
  dist.all_gather(chunks, x, group=group)
  return torch.cat(chunks, dim=dim)


@functools.wraps(Kandinsky6AttnProcessor.__call__)
def __patch__Kandinsky6AttnProcessor__call__(
  self: Kandinsky6AttnProcessor,
  attn,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  rotary_emb: Optional[torch.Tensor] = None,
  rotary_emb_kv: Optional[torch.Tensor] = None,
  attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  original = _CP_PATCH_ORIGINALS.get(Kandinsky6AttnProcessor)
  if original is None or not _cp_is_armed(self):
    return original(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      rotary_emb=rotary_emb,
      rotary_emb_kv=rotary_emb_kv,
      attn_mask=attn_mask,
    )

  # The CP runtime arms *every* attention it can find, including the 4+4 text blocks that run
  # outside the cached loop and hold the complete prompt. Only the video self-attention has a
  # sharded query, key and value, so it is the only call that may hand the armed config to
  # `dispatch_attention_fn`; the rest run locally, which is exact because their query is replicated
  # or their KV is complete.
  if getattr(attn, _SELF_DISTRIBUTED_ATTR, False):
    return original(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      rotary_emb=rotary_emb,
      rotary_emb_kv=rotary_emb_kv,
      attn_mask=attn_mask,
    )

  saved_parallel_config = self._parallel_config
  self._parallel_config = None
  try:
    if getattr(attn, _KV_GATHER_ATTR, False):
      # `av_cross_attention`: audio queries (replicated) against video KV (sharded). Gather the
      # video activation and its rope back to the full sequence, then run the attention locally.
      if encoder_hidden_states is None:
        raise ValueError(
          "Kandinsky 6.0 audio->video cross-attention requires `encoder_hidden_states` (the video "
          "stream) to gather the video KV.")
      encoder_hidden_states = _cp_all_gather(encoder_hidden_states, dim=1, processor=self)
      if rotary_emb_kv is not None:
        rotary_emb_kv = _cp_all_gather(rotary_emb_kv, dim=0, processor=self)
    return original(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      rotary_emb=rotary_emb,
      rotary_emb_kv=rotary_emb_kv,
      attn_mask=attn_mask,
    )
  finally:
    self._parallel_config = saved_parallel_config


def _install_cp_processor_patches() -> None:
  """Install the CP-aware attention processor patch once per process."""
  global _ATTN_PROCESSOR_PATCHED
  if _ATTN_PROCESSOR_PATCHED:
    return
  _CP_PATCH_ORIGINALS[Kandinsky6AttnProcessor] = Kandinsky6AttnProcessor.__call__
  Kandinsky6AttnProcessor.__call__ = __patch__Kandinsky6AttnProcessor__call__
  _ATTN_PROCESSOR_PATCHED = True


def _mark_attention_roles(transformer: Kandinsky6Transformer3DModel) -> None:
  """Tag which fused-block attentions stay distributed and which need a KV gather.

  `__patch__Kandinsky6AttnProcessor__call__` reads these attributes: exactly one attention per
  block (`video_dec_block.self_attention`) keeps the armed `parallel_config`, and exactly one
  (`av_cross_attention`) rebuilds its KV from the gathered video stream.

  :param transformer: Kandinsky 6.0 transformer being parallelized.
  """
  for block in transformer.visual_transformer_blocks:
    block.video_dec_block.self_attention._cp_self_distributed = True
    block.av_cross_attention._cp_gather_kv = True


@ContextParallelismPlannerRegister.register("Kandinsky6Transformer3DModel")
class Kandinsky6ContextParallelismPlanner(ContextParallelismPlanner):
  r"""Hook-based context parallelism for Kandinsky 6.0 (Ulysses).

  Only the video stream is sharded:

  * `visual_transformer_blocks.0` splits `vis` once, after `visual_embeddings` patchified it
    (the 5-D latent grid feeds `visual_ropa_pos` and cannot be split at the root).
  * every block splits its own copy of `vis_rope`, a `(S, 1, head_dim // 2, 2, 2)` table that
    `forward` recomputes once and passes to every block.
  * the **last** block's output is gathered: `forward` reshapes the video stream back to
    `(B, T, H, W, C)` right after the loop, so the gather has to happen before that reshape and a
    block-output hook is the only module boundary available.

  The audio stream stays replicated: its self-attention is redundantly computed on each rank
  (exact, and only a few hundred tokens), `va_cross_attention` keeps a complete KV for free, and
  `av_cross_attention` gathers the video KV — see `__patch__Kandinsky6AttnProcessor__call__`.

  Because the gather sits on the last block, **CP with caching requires
  `Bn_compute_blocks >= 1`**: with the `DBCacheConfig` default of 0 the last block is skipped on a
  cached step (nothing gathers) and, on a computed step, it lands in the middle group whose residual
  subtraction would then mix a gathered tensor with a local one.

  Ring attention and `--ulysses-anything` are rejected: the video KV gather assumes the plain
  equipartition that Ulysses uses, and neither combination can be validated here.
  """

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | Kandinsky6Transformer3DModel] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert parallelism_config is not None, "ParallelismConfig must be provided."
    if parallelism_config.ring_size is not None and parallelism_config.ring_size > 1:
      raise NotImplementedError(
        "Kandinsky 6.0 Context Parallelism does not support Ring attention: the fused block's "
        "audio->video cross-attention gathers the video KV, which assumes the plain Ulysses "
        "equipartition. Use Ulysses (`--parallel ulysses`).")
    if parallelism_config.ulysses_anything:
      raise NotImplementedError(
        "Kandinsky 6.0 Context Parallelism does not support `--ulysses-anything`: the video KV "
        "gather assumes equally sized shards. Use the default Ulysses partition.")

    ulysses_size = parallelism_config.ulysses_size or 1
    tp_size = parallelism_config.tp_size or 1
    heads = transformer.config.model_dim // sum(transformer.config.axes_dims)
    if ulysses_size > 1 and (heads // tp_size) % ulysses_size != 0:
      raise ValueError(
        f"Kandinsky 6.0 has {heads} video attention heads, i.e. {heads // tp_size} per rank after "
        f"tp_size={tp_size}, which is not divisible by ulysses_size={ulysses_size}.")

    _install_cp_processor_patches()
    _mark_attention_roles(transformer)

    blocks = transformer.visual_transformer_blocks
    last_index = len(blocks) - 1
    if last_index < 1:
      raise ValueError(
        "Kandinsky 6.0 Context Parallelism needs at least two fused blocks: the video stream is "
        "split at block 0 and gathered on the last block's output, which would be the same module "
        "with a single block.")

    # Tell the cache layer that this planner cannot run with the `Bn_compute_blocks` default (0):
    # the gather lives on the last block, which the cached path only executes inside the Bn group.
    transformer._cp_requires_min_bn_blocks = 1

    _cp_plan: _ContextParallelModelPlan = {
      # Must stay first on block 0: the cached path replays this hook to re-derive the residual.
      "visual_transformer_blocks.0": {
        "vis": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      # `vis_rope` keeps its 5-D `(S, 1, head_dim // 2, 2, 2)` layout; only the token axis moves.
      "visual_transformer_blocks.*": {
        "vis_rope": _ContextParallelInput(split_dim=0, expected_dims=5),
      },
      # `forward` reshapes the video stream right after the loop, so the gather goes on the last
      # block's output; the audio stream (second slot) is replicated and stays untouched.
      f"visual_transformer_blocks.{last_index}": (_ContextParallelOutput(gather_dim=1,
                                                                         expected_dims=3), None),
    }
    logger.debug(f"Generated Kandinsky 6.0 CP plan: {_cp_plan}")
    return _cp_plan


@TensorParallelismPlannerRegister.register("Kandinsky6Transformer3DModel")
class Kandinsky6TensorParallelismPlanner(TensorParallelismPlanner):
  r"""Tensor-parallel planner for Kandinsky 6.0.

  Every fused block owns six attentions, all built from the same `Kandinsky6Attention` class with
  `to_query` / `to_key` / `to_value` / `out_layer`. The columns are sharded on a shared grid: a
  cross-modal attention whose KV comes from the other stream (`va_cross_attention`: 4096-channel
  query against a 2048-channel audio KV, and the mirrored `av_cross_attention`) projects the other
  stream up to the local head range with the same `ColwiseParallel`, so query head *i* keeps
  pairing with KV head *i*. `num_heads` follows the shard because the processor reshapes with it.

  The two FFNs are GELU `FeedForward`s (`net.0.proj` column, `net.2` row), the 4+4 text blocks get
  the same treatment, and every `Kandinsky6Modulation` shards its weight while keeping a
  replicated output — the blocks `chunk(3)` / `split([4096, 4096, 2048])` it, which needs the full
  width. Embeddings, the time embeddings, the norms and both output heads stay replicated: they
  either produce or consume the full feature width and are negligible next to the block stack.
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
  def attention_plan(prefix: str) -> Dict[str, ParallelStyle]:
    """Return the shard plan for one `Kandinsky6Attention`.

    :param prefix: Dotted path of the attention inside its block.
    :returns: The plan entries for that attention.
    """
    return {
      f"{prefix}.to_query": ColwiseParallel(),
      f"{prefix}.to_key": ColwiseParallel(),
      f"{prefix}.to_value": ColwiseParallel(),
      f"{prefix}.out_layer": RowwiseParallel(),
    }

  def parallelize_transformer(
    self,
    transformer: Kandinsky6Transformer3DModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_size = tp_mesh.size()
    head_dim = sum(transformer.config.axes_dims)
    layer_plans: List[Dict[str, ParallelStyle]] = []

    for block in transformer.visual_transformer_blocks:
      assert isinstance(block, Kandinsky6FusedTransformerDecoderBlock)
      layer_plan: Dict[str, ParallelStyle] = {}
      for attn_prefix in (
          "video_dec_block.self_attention",
          "video_dec_block.cross_attention",
          "audio_dec_block.self_attention",
          "audio_dec_block.cross_attention",
          "va_cross_attention",
          "av_cross_attention",
      ):
        attention = block.get_submodule(attn_prefix)
        if attention.num_heads % tp_size != 0:
          raise ValueError(f"[TP] Unsupported tp_size={tp_size}: {attn_prefix} has "
                           f"{attention.num_heads} heads.")
        layer_plan.update(self.attention_plan(attn_prefix))
        # The processor reshapes with `attn.num_heads` and `head_dim` is model config.
        shard_div_attr(attention, "num_heads", tp_size)

      for stream in ("video_dec_block", "audio_dec_block"):
        layer_plan[f"{stream}.feed_forward.net.0.proj"] = ColwiseParallel()
        layer_plan[f"{stream}.feed_forward.net.2"] = RowwiseParallel()
        # AdaLN modulation is consumed with `chunk(3)`: shard the weight, replicate the output.
        layer_plan[f"{stream}.visual_modulation.out_layer"] = ColwiseParallel(
          output_layouts=Replicate())
      layer_plan["va_modulation.out_layer"] = ColwiseParallel(output_layouts=Replicate())
      layer_plan["av_modulation.out_layer"] = ColwiseParallel(output_layouts=Replicate())

      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    for stack_name in ("video_text_transformer_blocks", "audio_text_transformer_blocks"):
      for block in getattr(transformer, stack_name):
        layer_plan = self.attention_plan("attn")
        layer_plan["feed_forward.net.0.proj"] = ColwiseParallel()
        layer_plan["feed_forward.net.2"] = RowwiseParallel()
        layer_plan["text_modulation.out_layer"] = ColwiseParallel(output_layouts=Replicate())
        if block.attn.num_heads % tp_size != 0:
          raise ValueError(f"[TP] Unsupported tp_size={tp_size}: {stack_name} attention has "
                           f"{block.attn.num_heads} heads.")
        shard_div_attr(block.attn, "num_heads", tp_size)
        parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
        layer_plans.append(layer_plan)

    first_block = transformer.visual_transformer_blocks[0]
    local_q_rows = first_block.video_dec_block.self_attention.to_query.weight.to_local().shape[0]
    expected = (transformer.config.model_dim // head_dim // tp_size) * head_dim
    if local_q_rows != expected:
      raise ValueError(f"Sharded video Q projection has {local_q_rows} rows per rank, expected "
                       f"{expected}.")
    return transformer, layer_plans
