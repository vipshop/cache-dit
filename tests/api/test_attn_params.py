"""CLI coverage for `--attn-params`."""

import pytest

from cache_dit.attention.backends import ffpa
from cache_dit.cli import utils as cli_utils
from cache_dit.cli.params import ATTN_BACKENDS, ATTN_PARAM_SPECS, FFPA_BITS_BACKENDS

parser = cli_utils.get_args(parse=False)


def resolve(argv):
  return cli_utils.maybe_postprocess_args(parser.parse_args(argv))


def test_attn_flag_is_replaced_by_the_type_key() -> None:
  with pytest.raises(SystemExit):
    parser.parse_args(["--attn", "flash"])


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
  args = resolve(["--attn-params", f"type={backend},{text}"])
  assert args.attn == backend
  assert args.attn_fields == expected
  assert dict(calls) == expected


@pytest.mark.parametrize("backend", [b for b in ATTN_BACKENDS if b not in ATTN_PARAM_SPECS])
def test_backend_without_options_is_selected_by_type_alone(backend) -> None:
  args = resolve(["--attn-params", f"type={backend}"])
  assert (args.attn, args.attn_fields) == (backend, {})


@pytest.mark.parametrize(
  "text, backend",
  [
    ("ffpa,bits=16", "ffpa"),
    ("ffpa,bits=8", "ffpa_fp8"),
    ("ffpa,bits=4", "ffpa_fp4"),
    ("FFPA,bits=8", "ffpa_fp8"),
  ],
)
def test_ffpa_bits_selects_the_precision_family(text, backend) -> None:
  args = resolve(["--attn-params", text])
  assert (args.attn, args.attn_fields) == (backend, {})


def test_ffpa_bits_options_follow_the_selected_family(monkeypatch) -> None:
  calls = []
  for key in ("fp8_hadamard", "fp4_hadamard", "fp4_pv_mm_type"):
    monkeypatch.setattr(ffpa, f"set_ffpa_{key}", lambda value, key=key: calls.append((key, value)))
  args = resolve(["--attn-params", "ffpa,bits=8,fp8_hadamard=true"])
  assert (args.attn, args.attn_fields) == ("ffpa_fp8", {"fp8_hadamard": True})
  args = resolve(["--attn-params", "ffpa,bits=4,fp4_hadamard=true,fp4_pv_mm_type=fp8"])
  assert (args.attn, args.attn_fields) == ("ffpa_fp4", {
    "fp4_hadamard": True,
    "fp4_pv_mm_type": "fp8"
  })
  assert sorted(calls) == sorted([("fp8_hadamard", True), ("fp4_hadamard", True),
                                  ("fp4_pv_mm_type", "fp8")])


def test_ffpa_bits_shorthand_is_idempotent() -> None:
  once = resolve(["--attn-params", "ffpa,bits=8"])
  twice = cli_utils.maybe_postprocess_args(once)
  assert (twice.attn, twice.attn_fields) == ("ffpa_fp8", {})
  assert twice.attn_params == {"type": "ffpa", "bits": "8"}


@pytest.mark.parametrize(
  "text, full_text, suffix",
  [
    ("ffpa,bits=16,hybrid=true", "type=ffpa,hybrid=true", "_ffpa_hybrid"),
    ("ffpa,bits=8,fp8_hadamard=true", "type=ffpa_fp8,fp8_hadamard=true", "_ffpa_fp8_fp8_hadamard"),
    ("ffpa,bits=4,fp4_pv_mm_type=fp8", "type=ffpa_fp4,fp4_pv_mm_type=fp8",
     "_ffpa_fp4_fp4_pv_mm_type_fp8"),
  ],
)
def test_ffpa_bits_shorthand_and_full_name_strify_identically(monkeypatch, text, full_text,
                                                              suffix) -> None:
  for key in ("hybrid", "fp8_hadamard", "fp4_pv_mm_type"):
    monkeypatch.setattr(ffpa, f"set_ffpa_{key}", lambda value: None)
  shorthand = resolve(["--attn-params", text])
  full = resolve(["--attn-params", full_text])
  assert cli_utils.strify(shorthand, {}) == cli_utils.strify(full, {})
  assert cli_utils.strify(shorthand, {}).endswith(suffix)


def test_type_is_case_insensitive() -> None:
  assert resolve(["--attn-params", "type=Flash"]).attn == "flash"


def test_backend_list_covers_every_backend_with_options() -> None:
  assert set(ATTN_PARAM_SPECS) <= set(ATTN_BACKENDS)
  assert set(FFPA_BITS_BACKENDS.values()) <= set(ATTN_BACKENDS)


@pytest.mark.parametrize(
  "argv, message",
  [
    (["--attn-params", "type=flash,hybrid=true"], "takes no --attn-params options"),
    (["--attn-params", "type=ffpa_fp4,fp8_hadamard=true"], "Unknown --attn-params keys"),
    (["--attn-params", "type=ffpa,fp4_hadamard=true"], "Unknown --attn-params keys"),
    (["--attn-params", "type=ffpa_fp4,hybrid_n_early=100"], "multiple of 128"),
    (["--attn-params", "type=ffpa_fp4,fp4_pv_mm_type=fp16"], "Invalid --attn-params"),
    (["--attn-params", "ffpa,bits=7"], "Invalid --attn-params 'bits=7'"),
    (["--attn-params", "ffpa,bits="], "Invalid --attn-params 'bits='"),
    (["--attn-params", "ffpa,bits=fp8"], "Invalid --attn-params 'bits=fp8'"),
    (["--attn-params", "type=ffpa_fp8,bits=8"], "ffpa-family shorthand"),
    (["--attn-params", "type=flash,bits=8"], "ffpa-family shorthand"),
    (["--attn-params", "type=ffpa_fp8_per_block,bits=8"], "no per_block shorthand"),
    (["--attn-params", "ffpa,bits=4,fp8_hadamard=true"], "Unknown --attn-params keys"),
    (["--attn-params", "ffpa,bits=8,fp4_smooth_v=true"], "Unknown --attn-params keys"),
  ],
)
def test_attn_params_reject_wrong_backend_and_bad_values(argv, message) -> None:
  with pytest.raises(ValueError, match=message):
    resolve(argv)


@pytest.mark.parametrize("value, message", [
  ("", "needs at least one"),
  (" ", "needs at least one"),
  ("fp8_hadamard=true", "needs a 'type' key"),
  ("type=", "Invalid --attn-params 'type='"),
  ("type=nope", "Invalid --attn-params 'type=nope'"),
])
def test_bad_type_is_rejected_by_argparse(value, message, capsys) -> None:
  with pytest.raises(SystemExit):
    parser.parse_args(["--attn-params", value])
  assert message in capsys.readouterr().err


def test_attn_params_are_reflected_in_the_filename_in_a_fixed_order(monkeypatch) -> None:
  # The real setters keep module-level state; keep this test from leaking into other ffpa tests.
  for key in ("hybrid", "hybrid_n_early", "fp4_hadamard", "fp4_pv_mm_type", "fp4_smooth_v"):
    monkeypatch.setattr(ffpa, f"set_ffpa_{key}", lambda value: None)
  text = "fp4_smooth_v=true,fp4_pv_mm_type=fp8,hybrid=true,fp4_hadamard=true,hybrid_n_early=256"
  args = resolve(["--attn-params", f"type=ffpa_fp4,{text}"])
  assert cli_utils.strify(
    args,
    {}).endswith("ffpa_fp4_hybrid_n_early_256_hybrid_fp4_hadamard_fp4_pv_mm_type_fp8_fp4_smooth_v")
  assert cli_utils.strify(resolve(["--attn-params", "type=ffpa_fp4"]), {}).endswith("ffpa_fp4")
  assert resolve([]).attn is None
  assert resolve([]).attn_fields == {}


@pytest.mark.parametrize(
  "flag",
  [
    "--attn", "--sol-attn-tau", "--sol-attn-thresh-type", "--sol-attn-strict",
    "--sol-attn-sink-tokens", "--ffpa-hybrid", "--ffpa-hybrid-n-early", "--ffpa-fp4-hadamard",
    "--ffpa-fp8-hadamard", "--ffpa-fp4-pv-mm-type", "--ffpa-fp4-smooth-v", "--ffpa-fp8-smooth-v"
  ],
)
def test_removed_attention_flags_are_rejected(flag) -> None:
  with pytest.raises(SystemExit):
    parser.parse_args([flag, "x"])
