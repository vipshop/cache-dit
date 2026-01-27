import torch
import torch.distributed as dist


def send_tensor(
    tensor: torch.Tensor,
    dst: int,
    group: dist.ProcessGroup,
) -> None:
    tensor = tensor.contiguous()
    dist.send_object_list([tensor.shape], dst=dst, group=group, use_batch=True)
    dist.batch_isend_irecv([dist.P2POp(dist.isend, tensor, dst, group=group)]).pop().wait()


def recv_tensor(
    src: int,
    group: dist.ProcessGroup,
    device=None,
    dtype=None,
) -> torch.Tensor:
    objects = [None]
    dist.recv_object_list(objects, src=src, group=group, use_batch=True)
    t = torch.empty(objects[0], device=device, dtype=dtype)
    dist.batch_isend_irecv([dist.P2POp(dist.irecv, t, src, group=group)]).pop().wait()
    return t
