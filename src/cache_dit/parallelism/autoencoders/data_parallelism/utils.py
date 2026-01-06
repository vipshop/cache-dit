import torch
import torch.distributed as dist


def send_tensor(
    tensor: torch.Tensor,
    dst: int,
    group: dist.ProcessGroup,
) -> None:
    tensor = tensor.contiguous()
    dist.send_object_list([tensor.shape], dst=dst, group=group)
    dist.send(tensor, dst=dst, group=group)


def recv_tensor(
    src: int,
    group: dist.ProcessGroup,
    device=None,
    dtype=None,
) -> torch.Tensor:
    objects = [None]
    dist.recv_object_list(objects, src=src, group=group)
    t = torch.empty(objects[0], device=device, dtype=dtype)
    dist.recv(t, src=src, group=group)
    return t
