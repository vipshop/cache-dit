from typing import Optional

import torch

from ..core import _All2AllComm


class AsyncUlyssesOrchestrator:
  """Coordinate async Q/K/V and output collectives via `_All2AllComm`."""

  __slots__ = ("comm", "_handles")

  def __init__(self, cp_config):
    self.comm = _All2AllComm(cp_config)
    self._handles = {}

  def _send(self, name: str, tensor: torch.Tensor):
    handle = getattr(self.comm, f"send_{name}")(tensor)
    self._handles[name] = handle
    return handle

  def send_q(self, query: torch.Tensor):
    return self._send("q", query)

  def send_k(self, key: torch.Tensor):
    return self._send("k", key)

  def send_v(self, value: torch.Tensor):
    return self._send("v", value)

  def wait_qkv(
    self,
    handles: Optional[dict[str, object]] = None,
    order: tuple[str, str, str] = ("q", "k", "v"),
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    handles = self._handles if handles is None else handles
    return tuple(handles[name].wait() for name in order)

  def send_output(self, out: torch.Tensor):
    return self.comm.send_o(out)


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
