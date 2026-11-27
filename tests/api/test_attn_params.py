"""CLI coverage for `--attn-params`."""

import pytest

from cache_dit.attention.backends import ffpa
from cache_dit.cli import utils as cli_utils

parser = cli_utils.get_args(parse=False)


def resolve(argv):
  return cli_utils.maybe_postprocess_args(parser.parse_args(argv))


@pytest.mark.parametrize(
  "backend, text, expected",
  [
    ("ffpa", "hybrid=true,hybrid_n_early=256", {
      "hybrid": True,
      "hybrid_n_early": 256
    }),
    ("ffpa_fp8", "fp8_hadamard=true", {
      "fp8_hadamard": True
    }),
    ("ffpa_fp8_per_block", "fp8_smooth_v=true,hybrid=true", {
      "fp8_smooth_v": True,
      "hybrid": True
    }),
    (
      "ffpa_fp4",
      "fp4_pv_mm_type=fp8,fp4_hadamard=true",
      {
        "fp4_pv_mm_type": "fp8",
        "fp4_hadamard": True
      },
    ),
  ],
)
def test_ffpa_params_reach_the_backend_setters(monkeypatch, backend, text, expected) -> None:
  calls = []
  for key in expected:
    monkeypatch.setattr(ffpa, f"set_ffpa_{key}", lambda value, key=key: calls.append((key, value)))
  args = resolve(["--attn", backend, "--attn-params", text])
  assert args.attn_fields == expected
  assert dict(calls) == expected


@pytest.mark.parametrize(
  "argv, message",
  [
    (["--attn-params", "fp8_hadamard=true"], "requires --attn"),
    (["--attn", "flash", "--attn-params", "hybrid=true"], "takes no --attn-params"),
    (["--attn", "ffpa_fp4", "--attn-params", "fp8_hadamard=true"], "Unknown --attn-params keys"),
    (["--attn", "ffpa", "--attn-params", "fp4_hadamard=true"], "Unknown --attn-params keys"),
    (["--attn", "ffpa_fp4", "--attn-params", "hybrid_n_early=100"], "multiple of 128"),
    (["--attn", "ffpa_fp4", "--attn-params", "fp4_pv_mm_type=fp16"], "Invalid --attn-params"),
  ],
)
def test_attn_params_reject_wrong_backend_and_bad_values(argv, message) -> None:
  with pytest.raises(ValueError, match=message):
    resolve(argv)


def test_attn_params_are_reflected_in_the_filename_in_a_fixed_order(monkeypatch) -> None:
  # The real setters keep module-level state; keep this test from leaking into other ffpa tests.
  for key in ("hybrid", "hybrid_n_early", "fp4_hadamard", "fp4_pv_mm_type", "fp4_smooth_v"):
    monkeypatch.setattr(ffpa, f"set_ffpa_{key}", lambda value: None)
  text = "fp4_smooth_v=true,fp4_pv_mm_type=fp8,hybrid=true,fp4_hadamard=true,hybrid_n_early=256"
  args = resolve(["--attn", "ffpa_fp4", "--attn-params", text])
  assert cli_utils.strify(
    args,
    {}).endswith("ffpa_fp4_hybrid_n_early_256_hybrid_fp4_hadamard_fp4_pv_mm_type_fp8_fp4_smooth_v")
  assert cli_utils.strify(resolve(["--attn", "ffpa_fp4"]), {}).endswith("ffpa_fp4")
  assert resolve([]).attn_fields == {}


@pytest.mark.parametrize(
  "flag",
  [
    "--sol-attn-tau", "--sol-attn-thresh-type", "--sol-attn-strict", "--sol-attn-sink-tokens",
    "--ffpa-hybrid", "--ffpa-hybrid-n-early", "--ffpa-fp4-hadamard", "--ffpa-fp8-hadamard",
    "--ffpa-fp4-pv-mm-type", "--ffpa-fp4-smooth-v", "--ffpa-fp8-smooth-v"
  ],
)
def test_removed_attention_flags_are_rejected(flag) -> None:
  with pytest.raises(SystemExit):
    parser.parse_args([flag])
