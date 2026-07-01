import torch

from diffusers.models.transformers.transformer_ernie_image import (
  ErnieImageTransformer2DModel,
  ErnieImageTransformer2DModelOutput,
)
from .functor_base import PatchFunctor
from ...logger import init_logger

logger = init_logger(__name__)


class ErnieImagePatchFunctor(PatchFunctor):
  """Patch ErnieImageTransformer2DModel.forward to move ``temb`` assignment outside the ``for layer
  in self.layers`` loop.

  The original forward reassigns ``temb`` inside the loop body, which works
  with the native ``ModuleList`` iteration.  After :meth:`CacheAdapter.apply`
  replaces ``self.layers`` with ``UnifiedBlocks``, that reassignment may not
  execute as expected.  This patch moves the list construction *before* the
  loop so that the cached block wrappers always receive the full ``temb``.
  """

  def _apply(
    self,
    transformer: ErnieImageTransformer2DModel,
    **kwargs,
  ) -> ErnieImageTransformer2DModel:
    transformer.forward = _patched_forward.__get__(transformer)
    transformer._is_patched = True
    return transformer


# ---------------------------------------------------------------------------
# Patched forward (identical to original except ``temb`` moved before loop)
# ---------------------------------------------------------------------------


def _patched_forward(
  self: ErnieImageTransformer2DModel,
  hidden_states: torch.Tensor,
  timestep: torch.Tensor,
  text_bth: torch.Tensor,
  text_lens: torch.Tensor,
  return_dict: bool = True,
):
  device, dtype = hidden_states.device, hidden_states.dtype
  B, C, H, W = hidden_states.shape
  p, Hp, Wp = self.patch_size, H // self.patch_size, W // self.patch_size
  N_img = Hp * Wp

  img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
  if self.text_proj is not None and text_bth.numel() > 0:
    text_bth = self.text_proj(text_bth)
  Tmax = text_bth.shape[1]
  text_sbh = text_bth.transpose(0, 1).contiguous()

  x = torch.cat([img_sbh, text_sbh], dim=0)
  S = x.shape[0]

  # Position IDs
  text_ids = (torch.cat(
    [
      torch.arange(Tmax, device=device, dtype=torch.float32).view(1, Tmax, 1).expand(B, -1, -1),
      torch.zeros((B, Tmax, 2), device=device),
    ],
    dim=-1,
  ) if Tmax > 0 else torch.zeros((B, 0, 3), device=device))
  grid_yx = torch.stack(
    torch.meshgrid(
      torch.arange(Hp, device=device, dtype=torch.float32),
      torch.arange(Wp, device=device, dtype=torch.float32),
      indexing="ij",
    ),
    dim=-1,
  ).reshape(-1, 2)
  image_ids = torch.cat(
    [
      text_lens.float().view(B, 1, 1).expand(-1, N_img, -1),
      grid_yx.view(1, N_img, 2).expand(B, -1, -1)
    ],
    dim=-1,
  )
  rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

  valid_text = (torch.arange(Tmax, device=device).view(1, Tmax) < text_lens.view(B, 1)
                if Tmax > 0 else torch.zeros((B, 0), device=device, dtype=torch.bool))
  attention_mask = torch.cat([torch.ones((B, N_img), device=device, dtype=torch.bool), valid_text],
                             dim=1)[:, None, None, :]

  # AdaLN
  sample = self.time_proj(timestep)
  sample = sample.to(dtype=dtype)
  c = self.time_embedding(sample)
  shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
    t.unsqueeze(0).expand(S, -1, -1).contiguous() for t in self.adaLN_modulation(c).chunk(6, dim=-1)
  ]

  # FIX: move temb construction BEFORE the for-loop so that caching wrappers
  # (UnifiedBlocks) receive the full temb tuple regardless of loop structure.
  temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
  for layer in self.layers:
    if torch.is_grad_enabled() and self.gradient_checkpointing:
      x = self._gradient_checkpointing_func(
        layer,
        x,
        rotary_pos_emb,
        temb,
        attention_mask=attention_mask,
      )
    else:
      x = layer(x, rotary_pos_emb, temb, attention_mask=attention_mask)

  x = self.final_norm(x, c).type_as(x)
  patches = self.final_linear(x)[:N_img].transpose(0, 1).contiguous()
  output = (patches.view(B, Hp, Wp, p, p, self.out_channels).permute(0, 5, 1, 3, 2,
                                                                     4).contiguous().view(
                                                                       B, self.out_channels, H, W))

  return ErnieImageTransformer2DModelOutput(sample=output) if return_dict else (output, )
