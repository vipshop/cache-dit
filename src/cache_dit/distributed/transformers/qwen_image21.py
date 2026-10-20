import torch
from typing import Dict, List, Tuple

from diffusers import QwenImage21Transformer2DModel
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_qwenimage21 import (
  QwenImage21AttnProcessor,
  QwenImage21FlexAttnProcessor,
  QwenImage21KVLayerCache,
  QwenImage21TransformerBlock,
  _qwenimage21_prepare_qkv,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)
from ..config import ParallelismConfig
from ...logger import init_logger

logger = init_logger(__name__)

_ATTN_PROCESSOR_PATCHED = False


def _patch_attn_processor_for_cp() -> None:
  """Route the default attention processor through the replicated-prefix CP path.

  The layout follows sglang's native Qwen-Image 2.1 DiT and vllm-omni's `joint_strategy="front"`:
  the text/condition-image prefix is replicated on every rank and only the target image tokens are
  sharded, so the block-causal segments stay valid and never need reordering. Prefix rows are
  recomputed redundantly on each rank; target rows attend to `[prefix ; target gathered]`, which is
  exactly the full attention they get without CP.

  The gathered variant must not communicate twice: cache-dit installs `_parallel_config` on every
  processor, so each dispatch has to opt out with `parallel_config=None`.
  """

  global _ATTN_PROCESSOR_PATCHED
  if _ATTN_PROCESSOR_PATCHED:
    return

  original_call = QwenImage21AttnProcessor.__call__

  def cp_call(
    self,
    attn: "torch.nn.Module",
    hidden_states: torch.Tensor,
    attention_mask=None,
    rotary_emb=None,
    layer_cache: QwenImage21KVLayerCache = None,
    kv_cache_mode: str = None,
    cache_write_slice: slice = None,
    segments: list = None,
    key_valid: torch.Tensor = None,
    cp_meta: tuple = None,
  ) -> torch.Tensor:
    if cp_meta is None:
      return original_call(
        self,
        attn,
        hidden_states,
        attention_mask=attention_mask,
        rotary_emb=rotary_emb,
        layer_cache=layer_cache,
        kv_cache_mode=kv_cache_mode,
        cache_write_slice=cache_write_slice,
        segments=segments,
        key_valid=key_valid,
      )

    comm, q_prefix, kv_prefix = cp_meta
    query, key, value, seq_len_q = _qwenimage21_prepare_qkv(attn, hidden_states, rotary_emb,
                                                            layer_cache, kv_cache_mode,
                                                            cache_write_slice)

    outputs = []
    if q_prefix > 0:
      # Prefix rows are identical on every rank: reuse the block-causal segments as-is.
      for start, end, is_text in segments:
        seg_mask = None
        if is_text:
          seg_len = end - start
          seg_mask = torch.cat(
            [
              torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
              torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
            ],
            dim=1,
          )[None, None]
        if key_valid is not None:
          seg_key_valid = key_valid[:, None, None, :end]
          seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
        outputs.append(
          dispatch_attention_fn(
            query[:, start:end],
            key[:, :end],
            value[:, :end],
            attn_mask=seg_mask,
            dropout_p=0.0,
            backend=None,
            parallel_config=None,
          ))

    if seq_len_q > q_prefix:
      # Target rows: gather the target K/V of every rank, keep the replicated prefix in front.
      # K and V share one collective (`torch.stack` + gather on the sequence axis).
      kv_local = torch.stack([key[:, kv_prefix:], value[:, kv_prefix:]], dim=0)
      kv_full = comm.all_gather_tensor_dim(kv_local, dim=2)
      key_full, value_full = kv_full[0], kv_full[1]
      key_all = torch.cat([key[:, :kv_prefix], key_full], dim=1)
      value_all = torch.cat([value[:, :kv_prefix], value_full], dim=1)
      target_mask = None
      if key_valid is not None:
        target_mask = torch.cat(
          [
            key_valid[:, :kv_prefix],
            key_valid.new_ones(key_valid.shape[0], key_full.shape[1]),
          ],
          dim=1,
        )[:, None, None, :]
      outputs.append(
        dispatch_attention_fn(
          query[:, q_prefix:],
          key_all,
          value_all,
          attn_mask=target_mask,
          dropout_p=0.0,
          backend=None,
          parallel_config=None,
        ))

    hidden_states = torch.cat(outputs, dim=1)[:, :seq_len_q]
    hidden_states = hidden_states.flatten(2, 3).type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    return attn.to_out[1](hidden_states)

  QwenImage21AttnProcessor.__call__ = cp_call
  _ATTN_PROCESSOR_PATCHED = True


@ContextParallelismPlannerRegister.register("QwenImage21")
class QwenImage21ContextParallelismPlanner(ContextParallelismPlanner):
  """Context parallelism for Qwen-Image 2.1: replicated prefix, sharded target tokens.

  The joint sequence is `[text + condition images, target image]` and its attention is block-causal,
  where target rows already attend to the whole sequence. Sharding only the target tail therefore
  keeps every mask valid, and the prefix KV cache stays complete on every rank. The split and the
  gather live in `QwenImage21PatchFunctor` (a declarative hook cannot express a non-uniform
  boundary), so this planner returns an empty hook plan.

  Ring is rejected: it would split the block-causal mask. `ulysses_anything` is rejected for the
  same reason (uneven shards break the segment indices). `QwenImage21FlexAttnProcessor` is rejected
  because its flex `BlockMask` describes the full sequence and the flex kernel never communicates.
  """

  def _apply(
    self,
    transformer: QwenImage21Transformer2DModel,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Dict:
    if parallelism_config.ring_size is not None and parallelism_config.ring_size > 1:
      raise NotImplementedError(
        "Ring attention is not supported for Qwen-Image 2.1: it would split the block-causal mask. "
        "Use `--parallel ulysses` instead.")
    if parallelism_config.ulysses_anything:
      raise NotImplementedError(
        "ulysses_anything is not supported for Qwen-Image 2.1: the block-causal segments assume "
        "evenly sized target shards. Choose a cp_size that divides the target token count.")

    flex_blocks = [
      block for block in transformer.transformer_blocks
      if isinstance(block.attn.processor, QwenImage21FlexAttnProcessor)
    ]
    if flex_blocks:
      raise ValueError(
        "Qwen-Image 2.1 context parallelism requires `QwenImage21AttnProcessor`: the flex "
        "`BlockMask` path describes the full sequence and never communicates. Call "
        "`transformer.set_attn_processor(QwenImage21AttnProcessor())` before enabling parallelism.")

    if not getattr(transformer, "_is_patched", False):
      # The split/gather lives in the patch functor, so context parallelism needs it even when the
      # cache engine is not enabled. The functor is idempotent (`_is_patched`) for the cache path.
      from ...caching.patch_functors import QwenImage21PatchFunctor

      QwenImage21PatchFunctor().apply(transformer)

    _patch_attn_processor_for_cp()
    logger.info("Enabled Qwen-Image 2.1 context parallelism: replicated prefix, sharded target "
                f"tokens (ulysses_size={parallelism_config.ulysses_size}).")

    return {}


@TensorParallelismPlannerRegister.register("QwenImage21Transformer2DModel")
class QwenImage21TensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism for the single-stream Qwen-Image 2.1 transformer.

  Unlike Qwen-Image 1.x, this model must not shard `attn.heads`: its attention processor
  computes `head_dim = attn.inner_dim // attn.heads`, so a divided head count would multiply
  `head_dim` and break both the QK-norm (`RMSNorm(head_dim)`) and the 3-axis RoPE tables.
  `ColwiseParallel` keeps `use_local_output=True`, so `to_q` returns a local
  `inner_dim / tp` tensor that unflattens into `heads / tp` heads of the full head dim,
  which is what the processor already assumes. The shared `modulation`, `img_in`, `txt_in`,
  `norm_out` and `proj_out` stay replicated.
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

  def parallelize_transformer(
    self,
    transformer: QwenImage21Transformer2DModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_size = tp_mesh.size()
    layer_plans = []

    for _, block in transformer.transformer_blocks.named_children():
      assert isinstance(block, QwenImage21TransformerBlock)
      heads = block.attn.heads
      assert heads % tp_size == 0, (
        f"num_attention_heads ({heads}) must be divisible by tp_size ({tp_size}). "
        f"Available tp sizes: {[size for size in range(1, heads + 1) if heads % size == 0]}. "
        "`attn.heads` is intentionally kept at its full value for this model, so the local "
        "shard is inferred from `inner_dim // tp_size` inside the attention processor.")

      layer_plan = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
        "img_mlp.proj": ColwiseParallel(),
        "img_mlp.gate_layer": ColwiseParallel(),
        "img_mlp.out": RowwiseParallel(),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    return transformer, layer_plans
