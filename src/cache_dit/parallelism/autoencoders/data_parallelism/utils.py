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
        dist.send_object_list(
            [tensor.shape],
            dst=dst,
            group=group,
            device=self._s_comm_device,  # 'cpu' is more efficient
            use_batch=True,
        )
        send_op = dist.P2POp(dist.isend, tensor, dst, group=group)
        self._ops.append(send_op)

    def recv_tensor(
        self,
        src: int,
        group: dist.ProcessGroup,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        objects = [None]
        dist.recv_object_list(
            objects,
            src=src,
            group=group,
            device=self._s_comm_device,  # 'cpu' is more efficient
            use_batch=True,
        )
        t = torch.empty(objects[0], device=device, dtype=dtype)
        recv_op = dist.P2POp(dist.irecv, t, src, group=group)
        self._ops.append(recv_op)
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
