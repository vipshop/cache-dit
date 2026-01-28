from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoAttnProcessor

from cache_dit.logger import init_logger
from cache_dit.parallelism.config import ParallelismConfig

from .tp_plan_registers import TensorParallelismPlanner, TensorParallelismPlannerRegister
from .tp_utils import shard_divisible_attr

logger = init_logger(__name__)


class DistributedRMSNorm(nn.Module):
    def __init__(
        self,
        tp_mesh: DeviceMesh,
        normalized_shape: Union[int, list[int], torch.Size],
        eps: Optional[float],
        elementwise_affine: bool,
        weight: torch.nn.parameter.Parameter,
    ):
        super().__init__()
        self.tp_mesh = tp_mesh
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = normalized_shape
        self.eps = eps
        if self.elementwise_affine:
            assert weight is not None
        self.weight = weight

    @classmethod
    def from_rmsnorm(cls, tp_mesh: DeviceMesh, rmsnorm: nn.RMSNorm):
        assert len(rmsnorm.normalized_shape) == 1
        if rmsnorm.weight is not None:
            tp_size = tp_mesh.get_group().size()
            tp_rank = tp_mesh.get_group().rank()
            weight = rmsnorm.weight.chunk(tp_size, dim=0)[tp_rank]
        else:
            weight = None
        norm = cls(
            tp_mesh=tp_mesh,
            normalized_shape=rmsnorm.normalized_shape,
            eps=rmsnorm.eps,
            elementwise_affine=rmsnorm.elementwise_affine,
            weight=weight,
        )
        return norm

    def forward(self, x):
        if self.elementwise_affine:
            assert x.shape[-1] == self.weight.shape[0]
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        torch.distributed.all_reduce(
            mean_square,
            op=torch.distributed.ReduceOp.AVG,
            group=self.tp_mesh.get_group(),
        )
        root_mean_square = torch.sqrt(mean_square + self.eps)
        x_normed = x / root_mean_square
        if self.elementwise_affine:
            x_normed = x_normed * self.weight.to(device=x.device)
        assert x_normed.device.type != "cpu"
        return x_normed


class ShardRotaryEmbProcessor:
    """Shard query/key rotary embeddings to match TP-sharded heads/channels.

    - interleaved RoPE: cos/sin are (B, T, D) -> shard along last dim
    - split RoPE: cos/sin are (B, H, T, D/2) -> shard along head dim
    """

    def __init__(self, processor: LTX2AudioVideoAttnProcessor, tp_size: int, tp_rank: int):
        self.processor = processor
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    @classmethod
    def from_attn_processor(
        cls, processor: LTX2AudioVideoAttnProcessor, tp_size: int, tp_rank: int
    ) -> "ShardRotaryEmbProcessor":
        return cls(processor=processor, tp_size=tp_size, tp_rank=tp_rank)

    def _shard_rope(self, emb):
        if emb is None:
            return None
        cos, sin = emb
        if cos is None or sin is None:
            return emb
        # split rope: (B, H, T, D/2)
        if cos.ndim == 4:
            cos = torch.chunk(cos, self.tp_size, dim=1)[self.tp_rank]
            sin = torch.chunk(sin, self.tp_size, dim=1)[self.tp_rank]
        else:
            # interleaved rope: (B, T, D)
            cos = torch.chunk(cos, self.tp_size, dim=-1)[self.tp_rank]
            sin = torch.chunk(sin, self.tp_size, dim=-1)[self.tp_rank]
        return (cos, sin)

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        query_rotary_emb=None,
        key_rotary_emb=None,
    ) -> torch.Tensor:
        query_rotary_emb = self._shard_rope(query_rotary_emb)
        key_rotary_emb = self._shard_rope(key_rotary_emb)
        return self.processor(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            query_rotary_emb=query_rotary_emb,
            key_rotary_emb=key_rotary_emb,
        )


@TensorParallelismPlannerRegister.register("LTX2")
class LTX2VideoTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        return self.parallelize_transformer(transformer=transformer, tp_mesh=tp_mesh)

    def parallelize_transformer(self, transformer: nn.Module, tp_mesh: DeviceMesh):
        tp_size = tp_mesh.get_group().size()
        tp_rank = tp_mesh.get_group().rank()

        def _shard_attention(attn: nn.Module, what: str):
            shard_divisible_attr(
                attn,
                "heads",
                tp_size,
                what=what,
                context="LTX2VideoTensorParallelismPlanner",
            )

        def prepare_block(block: nn.Module):
            # Shard heads for all attention modules inside the block
            _shard_attention(block.attn1, "attn1")
            _shard_attention(block.attn2, "attn2")
            _shard_attention(block.audio_attn1, "audio_attn1")
            _shard_attention(block.audio_attn2, "audio_attn2")
            _shard_attention(block.audio_to_video_attn, "audio_to_video_attn")
            _shard_attention(block.video_to_audio_attn, "video_to_audio_attn")

            layer_plan = {
                # video self-attn / text cross-attn
                "attn1.to_q": ColwiseParallel(),
                "attn1.to_k": ColwiseParallel(),
                "attn1.to_v": ColwiseParallel(),
                "attn1.to_out.0": RowwiseParallel(),
                "attn2.to_q": ColwiseParallel(),
                "attn2.to_k": ColwiseParallel(),
                "attn2.to_v": ColwiseParallel(),
                "attn2.to_out.0": RowwiseParallel(),
                # audio self-attn / text cross-attn
                "audio_attn1.to_q": ColwiseParallel(),
                "audio_attn1.to_k": ColwiseParallel(),
                "audio_attn1.to_v": ColwiseParallel(),
                "audio_attn1.to_out.0": RowwiseParallel(),
                "audio_attn2.to_q": ColwiseParallel(),
                "audio_attn2.to_k": ColwiseParallel(),
                "audio_attn2.to_v": ColwiseParallel(),
                "audio_attn2.to_out.0": RowwiseParallel(),
                # a2v / v2a cross-attn
                "audio_to_video_attn.to_q": ColwiseParallel(),
                "audio_to_video_attn.to_k": ColwiseParallel(),
                "audio_to_video_attn.to_v": ColwiseParallel(),
                "audio_to_video_attn.to_out.0": RowwiseParallel(),
                "video_to_audio_attn.to_q": ColwiseParallel(),
                "video_to_audio_attn.to_k": ColwiseParallel(),
                "video_to_audio_attn.to_v": ColwiseParallel(),
                "video_to_audio_attn.to_out.0": RowwiseParallel(),
                # FFNs
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
                "audio_ff.net.0.proj": ColwiseParallel(),
                "audio_ff.net.2": RowwiseParallel(),
            }

            parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)

            # Shard qk norms
            for attn in (
                block.attn1,
                block.attn2,
                block.audio_attn1,
                block.audio_attn2,
                block.audio_to_video_attn,
                block.video_to_audio_attn,
            ):
                attn.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_q)
                attn.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_k)

        # Shard RoPE frequencies for all attention processors in every block.
        # NOTE: This assumes rotary embedding head counts align with attention heads (true for LTX-2 configs).
        for _, block in transformer.transformer_blocks.named_children():
            for attn_name in (
                "attn1",
                "attn2",
                "audio_attn1",
                "audio_attn2",
                "audio_to_video_attn",
                "video_to_audio_attn",
            ):
                attn = getattr(block, attn_name)
                if hasattr(attn, "processor") and isinstance(
                    attn.processor, LTX2AudioVideoAttnProcessor
                ):
                    attn.processor = ShardRotaryEmbProcessor.from_attn_processor(
                        processor=attn.processor,
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                    )
            prepare_block(block)

        return transformer
