import torch
from typing import Dict, List, Tuple

from diffusers import QwenImage21Transformer2DModel
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
from ...attention import _dispatch_attention_fn
from ...logger import init_logger

logger = init_logger(__name__)

_ATTN_PROCESSOR_PATCHED = False


def _patch_attn_processor_for_cp() -> None:
  """Run full-sequence Ulysses attention with head-sharded prefix KV caches.

  Q/K/V are projected on local sequence shards, then redistributed from
  `[B, S_LOCAL, H_GLOBAL, D]` to `[B, S_GLOBAL, H_LOCAL, D]`. Block-causal masks and segments keep
  their original global indices. The output all-to-all restores the block's local sequence layout.

  Every dispatch opts out with `parallel_config=None` because communication is explicit here.
  `cp_meta` contains `(communicator, global_sequence_padding, ulysses_degree)`.
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

    comm, cp_pad, cp_size = cp_meta
    query, key, value, _ = _qwenimage21_prepare_qkv(attn, hidden_states, rotary_emb, None, None,
                                                    None)
    local_heads = query.shape[2]
    if local_heads % cp_size != 0:
      raise NotImplementedError(
        f"Qwen-Image 2.1 CP requires the local attention head count after TP ({local_heads}) "
        f"to be divisible by ulysses_size ({cp_size}).")

    # Metadata contains the sequence length on some communication backends; prefill and cached
    # steps have different lengths, so refresh it for every attention call.
    comm.init_meta(query)
    query_wait = comm.send_q(query)
    key_wait = comm.send_k(key)
    value_wait = comm.send_v(value)
    query = query_wait.wait()
    key = key_wait.wait()
    value = value_wait.wait()

    seq_len = query.shape[1] - cp_pad
    query = query[:, :seq_len]
    key = key[:, :seq_len]
    value = value[:, :seq_len]

    if layer_cache is not None:
      if kv_cache_mode == "extract" and cache_write_slice is not None:
        layer_cache.store(key[:, cache_write_slice].clone(), value[:, cache_write_slice].clone())
      elif kv_cache_mode == "cached":
        cached_k, cached_v = layer_cache.get()
        key = torch.cat([cached_k, key], dim=1)
        value = torch.cat([cached_v, value], dim=1)

    if segments is None:
      hidden_states = _dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        backend=self._attention_backend,
        parallel_config=None,
      )
    else:
      prefix_len = segments[-1][1] if segments else 0
      outputs = []
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
          _dispatch_attention_fn(
            query[:, start:end],
            key[:, :end],
            value[:, :end],
            attn_mask=seg_mask,
            dropout_p=0.0,
            backend=None,
            parallel_config=None,
          ))
      outputs.append(
        _dispatch_attention_fn(
          query[:, prefix_len:],
          key,
          value,
          attn_mask=None if key_valid is None else key_valid[:, None, None, :],
          dropout_p=0.0,
          backend=None,
          parallel_config=None,
        ))
      hidden_states = torch.cat(outputs, dim=1)

    if cp_pad:
      pad_shape = list(hidden_states.shape)
      pad_shape[1] = cp_pad
      hidden_states = torch.cat([hidden_states, hidden_states.new_zeros(pad_shape)], dim=1)
    assert hidden_states.shape[1] % cp_size == 0
    hidden_states = comm.send_o(hidden_states).wait()
    hidden_states = hidden_states.flatten(2, 3).type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    return attn.to_out[1](hidden_states)

  QwenImage21AttnProcessor.__call__ = cp_call
  _ATTN_PROCESSOR_PATCHED = True


@ContextParallelismPlannerRegister.register("QwenImage21")
class QwenImage21ContextParallelismPlanner(ContextParallelismPlanner):
  """Context parallelism for Qwen-Image 2.1 with sharded prefix and target tokens.

  The current sequence is padded and sharded before every block. Attention redistributes Q/K/V to
  full-sequence/local-head layout, so block-causal segments keep their global indices and the prefix
  KV cache stores only this rank's heads. The split/gather lives in `QwenImage21PatchFunctor`, so
  this planner returns an empty hook plan.

  Ring is rejected: it would split the block-causal mask. `ulysses_anything` and flex attention are
  rejected because this model-specific processor implements the standard padded Ulysses path only.
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
        "ulysses_anything is not supported for Qwen-Image 2.1: use the standard padded Ulysses "
        "path with `--parallel ulysses` instead.")
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
    logger.info("Enabled Qwen-Image 2.1 context parallelism: full-sequence shards with "
                f"head-sharded prefix KV cache (ulysses_size={parallelism_config.ulysses_size}).")

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
