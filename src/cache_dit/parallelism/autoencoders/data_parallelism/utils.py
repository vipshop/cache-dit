import torch
import torch.distributed as dist
from cache_dit.platforms import current_platform


class TileBatchedP2PComm:
    def __init__(self):
        self._ops = []
        self._reqs = None
        self._comm_backend = dist.get_backend(dist.group.WORLD)
        self._s_comm_device = (
            "cpu" if "cpu" in self._comm_backend else current_platform.default_device()
        )

    def send_tensor(
        self,
        tensor: torch.Tensor,
        dst: int,
        group: dist.ProcessGroup,
    ) -> None:
        tensor = tensor.contiguous()
        s_dims = torch.tensor(len(tensor.shape), device=self._s_comm_device, dtype=torch.int64)
        s_shape = torch.tensor(tensor.shape, device=self._s_comm_device, dtype=torch.int64)
        send_op_d = dist.P2POp(dist.isend, s_dims, dst, group=group)
        send_op_s = dist.P2POp(dist.isend, s_shape, dst, group=group)
        dist.batch_isend_irecv([send_op_d]).pop().wait()
        dist.batch_isend_irecv([send_op_s]).pop().wait()

        send_op_t = dist.P2POp(dist.isend, tensor, dst, group=group)  # tile
        self._ops.append(send_op_t)

    def recv_tensor(
        self,
        src: int,
        group: dist.ProcessGroup,
        device=None,
        dtype=None,
    ) -> torch.Tensor:

        s_dims = torch.tensor(0, device=self._s_comm_device, dtype=torch.int64)
        recv_op_d = dist.P2POp(dist.irecv, s_dims, src, group=group)
        dist.batch_isend_irecv([recv_op_d]).pop().wait()

        s_shape = torch.empty((s_dims.item(),), device=self._s_comm_device, dtype=torch.int64)
        recv_op_s = dist.P2POp(dist.irecv, s_shape, src, group=group)
        dist.batch_isend_irecv([recv_op_s]).pop().wait()

        t = torch.empty(tuple(s_shape.tolist()), device=device, dtype=dtype)
        recv_op_t = dist.P2POp(dist.irecv, t, src, group=group)  # tile
        self._ops.append(recv_op_t)
        return t

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def sync(self):
        self.commit()
        self.wait()
