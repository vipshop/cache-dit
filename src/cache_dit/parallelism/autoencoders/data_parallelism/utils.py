import torch
import torch.distributed as dist
from typing import Optional, List
from cache_dit.platforms import current_platform


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
            s_shape = torch.empty((s_dims,), device=self._s_device, dtype=torch.int64)
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
