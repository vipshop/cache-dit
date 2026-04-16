from typing import Optional

import torch


def require_cp_config(instance, owner_name: str):
  cp_config = getattr(instance, "_cp_config", None)
  if cp_config is None:
    raise RuntimeError(f"{owner_name} is missing _cp_config during async Ulysses attention.")
  return cp_config


def maybe_wait(value):
  if isinstance(value, torch.Tensor):
    return value
  if hasattr(value, "wait"):
    return value.wait()
  if callable(value):
    return value()
  return value


def split_joint_hidden_states(
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
  if encoder_hidden_states is None:
    return None, hidden_states

  encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
    [
      encoder_hidden_states.shape[1],
      hidden_states.shape[1] - encoder_hidden_states.shape[1],
    ],
    dim=1,
  )
  return encoder_hidden_states, hidden_states


def flatten_attn_output(hidden_states: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
  if hidden_states.ndim == 4:
    hidden_states = hidden_states.flatten(2, 3)
  return hidden_states.to(dtype)
