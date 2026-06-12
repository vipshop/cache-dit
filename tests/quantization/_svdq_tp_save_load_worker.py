from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel
from torch.distributed.tensor.parallel import parallelize_module

from cache_dit.quantization.svdquant.dtensor import SVDQW4A4ShardLinear
from cache_dit.quantization.svdquant.ptq import _save_quantized_module
from cache_dit.quantization.svdquant.ptq import load_svdq
from cache_dit.quantization.svdquant.quantizer import _quantize_linear_svdq_w4a4_from_smooth_scale


class ToyModule(nn.Module):

  def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
    super().__init__()
    self.proj = nn.Linear(256, 256, bias=True, device=device, dtype=dtype)


def main() -> None:
  local_rank = int(os.environ["LOCAL_RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  checkpoint_path = Path(os.environ["CACHE_DIT_SVDQ_TP_SAVE_LOAD_PATH"])

  torch.cuda.set_device(local_rank)
  device = torch.device(f"cuda:{local_rank}")
  dtype = torch.bfloat16

  dist.init_process_group("nccl")
  mesh = init_device_mesh("cuda", (world_size, ))

  model = ToyModule(device, dtype).eval()
  parallelize_module(model, mesh, {"proj": ColwiseParallel()})

  smooth = torch.ones(256, device=device, dtype=dtype)
  quantized = _quantize_linear_svdq_w4a4_from_smooth_scale(
    model.proj,
    smooth,
    smooth_scale_orig=smooth,
    rank=32,
    precision="int4",
    torch_dtype=dtype,
    device=device,
  )
  assert isinstance(quantized, SVDQW4A4ShardLinear)
  model.proj = quantized

  writer_rank = _save_quantized_module(
    model,
    serialize_to=str(checkpoint_path),
    quant_type="svdq_int4_r32",
    rank=32,
    precision="int4",
    quantized_layer_names=["proj"],
    svdq_kwargs={"runtime_kernel": "v1"},
  )
  if local_rank == 0:
    assert writer_rank
    assert checkpoint_path.is_file()
  else:
    assert not writer_rank

  dist.barrier()

  loaded_model = ToyModule(device, dtype).eval()
  parallelize_module(loaded_model, mesh, {"proj": ColwiseParallel()})
  loaded_model = load_svdq(loaded_model, str(checkpoint_path))
  assert isinstance(loaded_model.proj, SVDQW4A4ShardLinear)

  original_state = model.proj.state_dict()
  loaded_state = loaded_model.proj.state_dict()
  for key in original_state:
    torch.testing.assert_close(loaded_state[key], original_state[key], rtol=0, atol=0)

  dist.barrier()
  dist.destroy_process_group()


if __name__ == "__main__":
  main()
