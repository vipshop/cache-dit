import subprocess
import sys

import pytest
import torch

from cache_dit.quantization import QuantizeConfig, quantize

pytest.importorskip("torchao")

# The torchao backend used to build an FP8 QuantizeAOContext as a default argument, so
# importing it raised the FP8 capability assertion on pre-Ada GPUs, whatever quant_type
# was requested.
_IMPORT_ON_SM80 = """
from cache_dit.platforms import current_platform
current_platform.get_device_capability = lambda *args, **kwargs: (8, 0)

from cache_dit.quantization.torchao.quantize_ao import QuantizeAOContext

QuantizeAOContext(quant_type="int8_per_row")
try:
  QuantizeAOContext(quant_type="float8_per_row")
except AssertionError:
  pass
else:
  raise SystemExit("float8_per_row was accepted on sm80")
"""


def test_torchao_backend_imports_on_pre_ada_gpu() -> None:
  # A fresh interpreter, since the failure happened at import time.
  result = subprocess.run([sys.executable, "-c", _IMPORT_ON_SM80], capture_output=True, text=True)
  assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="The torchao backend requires CUDA.")
@pytest.mark.parametrize("quant_type", ["int8_per_row", "int8_per_tensor"])
def test_int8_dynamic_quantization_matches_bf16(quant_type: str) -> None:
  torch.manual_seed(0)
  model = torch.nn.Sequential(
    torch.nn.Linear(256, 512),
    torch.nn.GELU(),
    torch.nn.Linear(512, 256),
  ).to("cuda", torch.bfloat16)
  x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
  ref = model(x)

  model = quantize(model, QuantizeConfig(quant_type=quant_type, regional_quantize=False))
  out = model(x)

  rel_err = (out.float() - ref.float()).norm() / ref.float().norm()
  assert rel_err < 5e-2
