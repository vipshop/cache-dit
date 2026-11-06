import torch
import torch.distributed as dist
from typing import Dict, List, Optional
from ...platforms import current_platform


class TileBatchedP2PComm:

  def __init__(self):
    self._ops: List[dist.P2POp] = []
    self._reqs: List[dist.Work] = []
    self._backend = dist.get_backend(dist.group.WORLD)
    # Use CPU for communication to avoid Host-GPU sync overhead
    self._s_device = "cpu" if "cpu" in self._backend else current_platform.default_device()
    # We can set s_dims and s_shape before sending/receiving tensors,
    # thus, can reduce the number of ops in each commit.
    # WARN: The set_xxx and clear_xxx methods must be called by all ranks
    # in order to avoid deadlock. The dims will always be the same across ranks,
    # but the shape may be different.
    self._s_dims: Optional[int] = None
    self._s_shape: Optional[torch.Size] = None

  def set_dims(self, dims: int):
    self._s_dims = dims

  def set_shape(self, shape: torch.Size):
    self._s_shape = shape

  def clear_dims(self):
    self._s_dims = None

  def clear_shape(self):
    self._s_shape = None

  def send_shape(
    self,
    tensor: torch.Tensor,
    dst: int,
    group: dist.ProcessGroup,
  ):
    if self._s_dims is None:
      s_dims = torch.tensor(tensor.dim(), device=self._s_device, dtype=torch.int64)
      send_op_d = dist.P2POp(dist.isend, s_dims, dst, group=group)
      dist.batch_isend_irecv([send_op_d]).pop().wait()

    if self._s_shape is None:
      s_shape = torch.tensor(tensor.shape, device=self._s_device, dtype=torch.int64)
      send_op_s = dist.P2POp(dist.isend, s_shape, dst, group=group)
      dist.batch_isend_irecv([send_op_s]).pop().wait()

  def recv_shape(
    self,
    src: int,
    group: dist.ProcessGroup,
  ) -> torch.Size:
    if self._s_dims is None:
      s_dims = torch.tensor(0, device=self._s_device, dtype=torch.int64)
      recv_op_d = dist.P2POp(dist.irecv, s_dims, src, group=group)
      dist.batch_isend_irecv([recv_op_d]).pop().wait()
      s_dims = s_dims.item()
    else:
      s_dims = self._s_dims

    if self._s_shape is None:
      s_shape = torch.empty((s_dims, ), device=self._s_device, dtype=torch.int64)
      recv_op_s = dist.P2POp(dist.irecv, s_shape, src, group=group)
      dist.batch_isend_irecv([recv_op_s]).pop().wait()
      s_shape = torch.Size(s_shape.tolist())
    else:
      s_shape = self._s_shape

    return s_shape

  def send_tensor(
    self,
    tensor: torch.Tensor,
    dst: int,
    group: dist.ProcessGroup,
  ) -> None:

    tensor = tensor.contiguous()
    self.send_shape(tensor, dst, group)
    send_op_t = dist.P2POp(dist.isend, tensor, dst, group=group)
    self._ops.append(send_op_t)

  def recv_tensor(
    self,
    src: int,
    group: dist.ProcessGroup,
    device=None,
    dtype=None,
  ) -> torch.Tensor:

    s_shape = self.recv_shape(src, group)
    t = torch.empty(s_shape, device=device, dtype=dtype)
    recv_op_t = dist.P2POp(dist.irecv, t, src, group=group)  # tile
    self._ops.append(recv_op_t)

    return t

  def commit(self):
    if len(self._reqs) > 0:
      raise RuntimeError("commit called twice")
    self._reqs.extend(dist.batch_isend_irecv(self._ops))

  def wait(self):
    if len(self._reqs) == 0:
      raise RuntimeError("wait called before commit")
    for req in self._reqs:
      req.wait()
    self._ops = []
    self._reqs = []

  def sync(self):
    self.commit()
    self.wait()


_DTYPE_CODES: Dict[torch.dtype, int] = {
  torch.float16: 0,
  torch.bfloat16: 1,
  torch.float32: 2,
  torch.float64: 3,
  torch.int8: 4,
  torch.uint8: 5,
  torch.int32: 6,
  torch.int64: 7,
}
_CODE_DTYPES = {code: dtype for dtype, code in _DTYPE_CODES.items()}


def _dtype_code(dtype: torch.dtype) -> int:
  """Encode a dtype for the wire.

  :param dtype: Tensor dtype.
  :returns: The code.
  """

  try:
    return _DTYPE_CODES[dtype]
  except KeyError as error:
    raise TypeError(f"TileBatchedP2PComm cannot announce the dtype {dtype}") from error


# `TileSafeP2PComm` completes one message at a time, which multi-tile phases need: a phase that
# sends shape, payload, shape, payload cannot be batched without letting a deferred payload be
# consumed by the next shape receive. Planners whose decode splits into several tiles use it;
# every other planner keeps `TileBatchedP2PComm`, which batches a phase and is faster.
class TileSafeP2PComm:
  """Point-to-point exchange of tiled tensors together with their shapes.

  Every backend gives exactly one FIFO per peer and nothing more: `nccl` **silently ignores `tag`**
  (measured here: `isend(tag=1)` followed by `isend(tag=0)` arrives in send order whatever tag the
  receive asks for), so "put shapes on their own tag" cannot separate the two channels. The only
  portable way to keep a shape from being confused with a payload is to make each message complete
  before the next one is posted:

      send_tensor(t) = send(shape) -> send(t)

  A previous version sent the shape immediately but deferred the payload to `commit()`. Two
  `recv_tensor` calls in a row -- what every tile loop does -- then posted two shape receives before
  either payload receive while the sender interleaved `shape, payload, shape, payload`, so the second
  shape receive consumed a payload: the receiver read float32 bits as a shape and `torch.empty` raised
  "SymIntArrayRef expected to contain only concrete integers" (on `gloo`: "Received data size doesn't
  match expected size", 480 vs 40 bytes).

  Messages are therefore synchronous. `commit`/`wait`/`sync` are kept as no-ops because every planner
  calls them between phases; with nothing ever left in flight they have nothing to do.
  """

  def __init__(self):
    self._ops: List[dist.P2POp] = []
    self._reqs: List[dist.Work] = []
    self._backend = dist.get_backend(dist.group.WORLD)
    # Use CPU for communication to avoid Host-GPU sync overhead. `nccl` has no CPU transport, so it
    # gets the accelerator; `gloo`/`mpi` keep the announcement tensors on the host.
    self._s_device = "cpu" if "nccl" not in str(
      self._backend) else current_platform.default_device()
    # The number of dimensions is identical on every rank (each planner sets it once), so it travels
    # once. The shape cannot be cached: it differs per tile, and a per-instance "already sent" flag
    # desynchronises the protocol for the rest of the run as soon as the two ranks disagree about it.
    self._s_dims: Optional[int] = None
    # Dtype of the last announced payload, set by `recv_shape`.
    self._last_dtype: Optional[torch.dtype] = None

  def set_dims(self, dims: int):
    self._s_dims = dims

  def clear_dims(self):
    self._s_dims = None

  def set_shape(self, shape: torch.Size):
    """Deprecated no-op: the shape is always transmitted.

    Forty bytes per tile is not worth a cache that can desynchronise the protocol when only one rank
    calls it.

    :param shape: Ignored.
    """

  def clear_shape(self):
    """Deprecated no-op, see :meth:`set_shape`."""

  def _send(self, tensor: torch.Tensor, dst: int, group: dist.ProcessGroup) -> None:
    """Send one tensor and wait for the transfer to be accepted.

    :param tensor: Payload; must stay alive until the call returns.
    :param dst: Destination rank.
    :param group: Process group.
    """

    dist.isend(tensor, dst, group=group).wait()

  def _recv(self, tensor: torch.Tensor, src: int, group: dist.ProcessGroup) -> None:
    """Receive one tensor into `tensor` and wait for it.

    :param tensor: Destination buffer.
    :param src: Source rank.
    :param group: Process group.
    """

    dist.irecv(tensor, src, group=group).wait()

  def send_shape(
    self,
    tensor: torch.Tensor,
    dst: int,
    group: dist.ProcessGroup,
  ):
    """Announce the rank count (once) and the shape, completing before the payload goes out.

    :param tensor: Tensor whose shape is announced.
    :param dst: Destination rank.
    :param group: Process group.
    """

    if self._s_dims is None:
      self._send(torch.tensor([tensor.dim()], device=self._s_device, dtype=torch.int64), dst, group)
    # One int64 tensor carries both the shape and the dtype code: a separate dtype message would add
    # another blocking round trip per tile, which showed up as ~8% of the decode phase.
    self._send(
      torch.tensor([*tensor.shape, _dtype_code(tensor.dtype)],
                   device=self._s_device,
                   dtype=torch.int64), dst, group)

  def recv_shape(
    self,
    src: int,
    group: dist.ProcessGroup,
  ) -> torch.Size:
    """Receive the announced shape, blocking until it is here.

    Blocking is required, not stylistic: the payload buffer is allocated from the shape.

    :param src: Source rank.
    :param group: Process group.
    :returns: The peer's tensor shape.
    """

    if self._s_dims is None:
      s_dims = torch.empty(1, device=self._s_device, dtype=torch.int64)
      self._recv(s_dims, src, group)
      dims = int(s_dims.item())
    else:
      dims = self._s_dims

    announced = torch.empty((dims + 1, ), device=self._s_device, dtype=torch.int64)
    self._recv(announced, src, group)
    self._last_dtype = _CODE_DTYPES[int(announced[-1].item())]
    return torch.Size(announced[:-1].tolist())

  def send_tensor(
    self,
    tensor: torch.Tensor,
    dst: int,
    group: dist.ProcessGroup,
  ) -> None:

    tensor = tensor.contiguous()
    self.send_shape(tensor, dst, group)
    self._send(tensor, dst, group)

  def recv_tensor(
    self,
    src: int,
    group: dist.ProcessGroup,
    device=None,
    dtype=None,
  ) -> torch.Tensor:

    s_shape = self.recv_shape(src, group)
    # The announced dtype wins: the caller's guess is what used to desynchronise `nccl` (bfloat16 tile
    # against a float32 receive), and the sender is the only side that knows what it put on the wire.
    dtype = self._last_dtype or dtype
    t = torch.empty(s_shape, device=device, dtype=dtype)
    self._recv(t, src, group)

    return t

  def commit(self):
    """No-op: every message already completed when it was posted."""

  def wait(self):
    """No-op: nothing is ever left in flight."""

  def sync(self):
    """No-op: kept because every planner separates its phases with it."""
