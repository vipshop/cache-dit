import importlib.util

if importlib.util.find_spec("setuptools_scm") is None:
  raise ImportError("setuptools-scm is not installed. Install it by `pip install setuptools-scm`")

import os
import re
import subprocess
import warnings
from collections.abc import Callable
from os import path
from pathlib import Path

from setuptools import find_packages, setup
from setuptools_scm.version import get_local_dirty_tag

try:
  from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:
  _bdist_wheel = None

ROOT_DIR = Path(__file__).resolve().parent
RELEASE_FLAVOR_ENV = "CACHE_DIT_RELEASE_FLAVOR"
VERSION_WRITE_TO_ENV = "CACHE_DIT_VERSION_WRITE_TO"
CUDA_BUILD_FLAG = "CACHE_DIT_BUILD_CUDA"
CUDA_BUILD_MODE_ENV = "CACHE_DIT_CUDA_BUILD_MODE"
WHEEL_PLATFORM_TAG_ENV = "CACHE_DIT_WHEEL_PLATFORM_TAG"
# Distribution name is intentionally not managed via a PACKAGE_NAME env var in
# setup.py. Under PEP 621, setuptools reads the project name from the active
# workspace's pyproject.toml. Default builds therefore use the repository root
# pyproject.toml, and so do the release builds: the pure-Python wheel and the CUDA
# wheel share the distribution name, and only the CUDA wheel's local version
# (+cu13) tells them apart. tools/release_workspace.py prepares the workspace the
# release build runs in.
SPDLOG_SUBMODULE_PATH = ROOT_DIR / "csrc" / "third_party" / "spdlog"
SPDLOG_HEADER_PATH = SPDLOG_SUBMODULE_PATH / "include" / "spdlog" / "spdlog.h"
# Default builds cover Ada (89), Hopper architecture-specific (90a), Blackwell
# datacenter (100), and Blackwell consumer architecture-specific (120a). The single
# `_C_ops` extension filters out Hopper because its SVDQuant INT4 kernels exclude sm90;
# its FP4 kernels still require sm120a or newer at runtime.
DEFAULT_CUDA_ARCH_TARGETS = ("89", "90a", "100", "120a")
CUDA_ARCH_ALIASES = {
  "maxwell": "50",
  "pascal": "60",
  "volta": "70",
  "turing": "75",
  "ampere": "80",
  "ada": "89",
  "hopper": "90",
  "blackwell": "100",
}


def _normalize_sm_target(sm: str, *, support_sm120: bool, support_sm121: bool) -> str:
  if sm == "120" and support_sm120:
    return "120a"
  if sm == "121" and support_sm121:
    return "121a"
  return sm


def _default_cuda_arch_targets(
  *,
  support_sm90a: bool,
  support_sm100: bool,
  support_sm120: bool,
) -> list[str]:
  """Return the default CUDA targets supported by the active nvcc toolchain.

  :param support_sm90a: Whether nvcc supports the Hopper architecture-specific target.
  :param support_sm100: Whether nvcc supports the Blackwell datacenter target.
  :param support_sm120: Whether nvcc supports the Blackwell consumer architecture-specific
    target.
  :returns: The supported subset of `DEFAULT_CUDA_ARCH_TARGETS`.
  """

  support = {
    "89": True,
    "90a": support_sm90a,
    "100": support_sm100,
    "120a": support_sm120,
  }
  return [sm for sm in DEFAULT_CUDA_ARCH_TARGETS if support.get(sm, False)]


def _is_supported_cuda_sm_target(sm: str) -> bool:
  """Return whether a normalized SM target can host the single `cache_dit._C_ops` extension.

  Every kernel family now shares one extension, so this predicate is the intersection of the
  per-family floors: the communication and attention kernels need `>=sm89`, and the SVDQuant
  INT4 kernels exclude Hopper (`sm90`). A Hopper-only target list therefore resolves to an
  empty set and the extension is skipped as a whole.

  :param sm: Normalized SM target such as `89`, `90`, or `120a`.
  :returns: `True` when the target is `>=sm89` and is not `sm90`.
  """

  normalized = sm.removesuffix("a")
  if not normalized.isdigit():
    return False

  capability = int(normalized)
  return capability >= 89 and capability != 90


def _env_flag(name: str) -> bool:
  return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _get_cutlass_include_dir() -> str:
  """Resolve CUTLASS 4.6.1, defaulting to the repository submodule.

  `csrc/third_party/cutlass` is the reproducible default. CACHE_DIT_CUTLASS_DIR is an explicit
  override for development only; no sibling checkout or machine-specific path is assumed.

  :returns: Absolute include directory containing `cute/tensor.hpp`.
  :raises RuntimeError: If the submodule/override is missing or has the wrong version.
  """
  configured = os.getenv("CACHE_DIT_CUTLASS_DIR")
  candidates = [
    Path(configured) if configured else None, ROOT_DIR / "csrc" / "third_party" / "cutlass"
  ]
  for candidate in candidates:
    if candidate is None:
      continue
    include_dir = candidate if candidate.name == "include" else candidate / "include"
    if not (include_dir / "cute" / "tensor.hpp").is_file():
      continue
    version_file = include_dir / "cutlass" / "version.h"
    if not version_file.is_file():
      raise RuntimeError(f"Missing CUTLASS version header: {version_file}")
    version_text = version_file.read_text()
    versions = {}
    for component in ("MAJOR", "MINOR", "PATCH"):
      match = re.search(rf"#define CUTLASS_{component} (\d+)", version_text)
      if match is None:
        raise RuntimeError(f"Unable to parse CUTLASS_{component} from {version_file}")
      versions[component] = int(match.group(1))
    version = (versions["MAJOR"], versions["MINOR"], versions["PATCH"])
    if version != (4, 6, 1):
      raise RuntimeError(f"The SM120 CuTe SVDQ kernel requires CUTLASS 4.6.1, found {version}")
    return str(include_dir)
  raise RuntimeError(
    "The SM120 CuTe SVDQ kernel requires CUTLASS 4.6.1. Initialize the bundled dependency with "
    "`git submodule update --init --recursive`, or explicitly set CACHE_DIT_CUTLASS_DIR to a "
    "compatible checkout.")


def _get_release_flavor() -> str:
  flavor = os.getenv(RELEASE_FLAVOR_ENV, "base").strip().lower()
  if flavor not in {"base", "cu13"}:
    raise RuntimeError(f"Unsupported {RELEASE_FLAVOR_ENV}={flavor!r}. Expected 'base' or 'cu13'.")
  return flavor


RELEASE_FLAVOR = _get_release_flavor()
IS_CU13_RELEASE = RELEASE_FLAVOR == "cu13"


def _get_release_platform_tag() -> str:
  if not IS_CU13_RELEASE:
    return ""

  platform_tag = os.getenv(WHEEL_PLATFORM_TAG_ENV, "").strip()
  return platform_tag or "linux_x86_64"


def _should_build_cuda() -> bool:
  """Return whether the optional CUDA extension takes part in the build.

  `CACHE_DIT_BUILD_CUDA=1` is the single switch: the build asks for `cache_dit._C_ops`,
  which links every kernel family (SVDQuant, communication, attention, and model-specific)
  into one module, and the extension is skipped with a warning when none of the resolved SM
  targets can host it.

  :returns: `True` when the build requests the CUDA extension.
  """

  return _env_flag(CUDA_BUILD_FLAG)


def _get_cuda_build_mode() -> str:
  """Return the validated CUDA build mode.

  :returns: `FAST`, which compiles only the architectures detected on this machine, or
    `ALL`, which compiles the default architecture set.
  :raises RuntimeError: If the value is neither `FAST` nor `ALL`.
  """

  build_mode = os.getenv(CUDA_BUILD_MODE_ENV, "ALL").strip().upper()
  if build_mode not in {"FAST", "ALL"}:
    raise RuntimeError(
      f"Unsupported {CUDA_BUILD_MODE_ENV}={build_mode!r}. Expected 'FAST' or 'ALL'.")
  return build_mode


class CacheDiTSkipWarning(UserWarning):
  """Warning emitted when part of a requested CUDA build cannot be compiled."""


def _warn_skipped(message: str) -> None:
  """Report a skipped extension or SM target without failing the build.

  `CACHE_DIT_BUILD_CUDA=1` promises to keep building whatever is left, so a skip must not
  be able to fail the build: the category is filtered to `always` here, which keeps
  `-W error` / `PYTHONWARNINGS=error` from turning it into an exception. A warning rather
  than a print, because pip forwards the build backend's warnings and drops its stdout.

  :param message: The skip notice to emit.
  """

  with warnings.catch_warnings():
    warnings.simplefilter("always", CacheDiTSkipWarning)
    warnings.warn(message, CacheDiTSkipWarning, stacklevel=2)


if _should_build_cuda():
  try:
    import torch
    from packaging import version as packaging_version
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
  except ImportError as exc:
    raise RuntimeError(
      "Building the optional Cache-DiT CUDA extensions requires `torch` and `packaging` "
      "to be installed in the active environment. Run `conda activate cdit` and "
      "initialize submodules with `git submodule update --init --recursive --force`, then install with "
      "`CACHE_DIT_BUILD_CUDA=1 python3 -m pip install -e . --no-build-isolation`.") from exc
else:
  torch = None
  packaging_version = None
  CUDA_HOME = None
  BuildExtension = None
  CUDAExtension = None

if BuildExtension is None:
  CacheDiTBuildExtension = None
else:

  class CacheDiTBuildExtension(BuildExtension):
    """Merge the per-toolchain compiler flag buckets before the extensions build.

    `CUDAExtension` only forwards the `cxx` and `nvcc` buckets to the compiler, so the
    `gcc`, `msvc`, and `nvcc_msvc` buckets declared by the extension definitions are
    folded into those two buckets here.
    """

    def build_extensions(self):
      for ext in self.extensions:
        ext.extra_compile_args.setdefault("cxx", [])
        ext.extra_compile_args.setdefault("nvcc", [])
        if self.compiler.compiler_type == "msvc":
          ext.extra_compile_args["cxx"] += ext.extra_compile_args.pop("msvc", [])
          ext.extra_compile_args["nvcc"] += ext.extra_compile_args.pop("nvcc_msvc", [])
        else:
          ext.extra_compile_args["cxx"] += ext.extra_compile_args.pop("gcc", [])
      super().build_extensions()


def _ensure_spdlog_submodule() -> None:
  if SPDLOG_HEADER_PATH.exists():
    return

  try:
    subprocess.check_call(
      ["git", "submodule", "update", "--init", "--recursive", "--force"],
      cwd=ROOT_DIR,
    )
  except (OSError, subprocess.CalledProcessError) as exc:
    raise RuntimeError(
      "The SVDQuant build requires the git submodule `csrc/third_party/spdlog`, "
      "and automatic submodule initialization failed. Run "
      "`git submodule update --init --recursive --force` manually, then rerun the install command."
    ) from exc

  if SPDLOG_HEADER_PATH.exists():
    return

  raise RuntimeError(
    "The SVDQuant build requires the git submodule `csrc/third_party/spdlog`, "
    "but the expected header is still missing after automatic submodule initialization. "
    "Run `git submodule update --init --recursive --force` manually and verify the submodule checkout, "
    "then rerun the install command.")


def _parse_arch_list(raw_arch_list: str) -> list[str]:
  arch_list: list[str] = []
  for arch in re.split(r"[;, ]+", raw_arch_list):
    normalized = arch.strip().lower()
    if not normalized:
      continue
    normalized = normalized.removesuffix("+ptx")
    normalized = normalized.removeprefix("sm_").removeprefix("compute_")
    normalized = normalized.replace(".", "")
    normalized = CUDA_ARCH_ALIASES.get(normalized, normalized)
    arch_list.append(normalized)
  return arch_list


def _required_svdquant_sources() -> list[str]:
  return [
    "csrc/kernels/svdq/svdq.cpp",
    "csrc/kernels/svdq/torch.cpp",
    # One native W4A4 GEMM host TU plus the explicit INT4 / NVFP4 instantiations.
    "csrc/kernels/svdq/native/gemm_w4a4.cu",
    "csrc/kernels/svdq/native/gemm_w4a4_int4.cu",
    "csrc/kernels/svdq/native/gemm_w4a4_fp4.cu",
    "csrc/kernels/svdq/cute/fp4/gemm/gemm_w4a4.cu",
    "csrc/kernels/svdq/cute/fp4/quantize/quantize_uni.cu",
    "csrc/kernels/svdq/cute/fp4/quantize/quantize_r32.cu",
    "csrc/kernels/svdq/cute/fp4/quantize/quantize_r64.cu",
    "csrc/kernels/svdq/cute/fp4/quantize/quantize_r128.cu",
  ]


def _required_comm_sources() -> list[str]:
  return [
    "csrc/kernels/comm/comm.cpp",
    "csrc/kernels/comm/fp8_comm.cu",
  ]


def _required_attn_sources() -> list[str]:
  return [
    "csrc/kernels/attn/attn.cpp",
    "csrc/kernels/attn/merge_states.cu",
  ]


def _required_models_sources() -> list[str]:
  return [
    "csrc/kernels/models/models.cpp",
    "csrc/kernels/models/boogu.cpp",
    "csrc/kernels/models/boogu_scaled_rms_norm.cu",
    "csrc/kernels/models/boogu_gated_rms_norm.cu",
    "csrc/kernels/models/boogu_qk_norm_rope.cu",
    "csrc/kernels/models/qwen_image21.cpp",
    "csrc/kernels/models/qwen_image21_qk_norm_rope.cu",
    "csrc/kernels/models/qwen_image21_modulate_affine.cu",
    "csrc/kernels/models/qwen_image21_gated_residual.cu",
    "csrc/kernels/models/qwen_image21_silu_mul.cu",
  ]


def _missing_models_sources() -> list[str]:
  return [
    relative_path for relative_path in _required_models_sources()
    if not (ROOT_DIR / relative_path).exists()
  ]


def _required_cuda_sources() -> list[str]:
  """Return every translation unit linked into `cache_dit._C_ops`.

  The per-family lists stay separate because `tests/kernels/test_svdquant_source_manifest.py`
  and `tests/kernels/test_models_source_manifest.py` extract them from this file with `ast`.
  The entry point is listed first: it is the only TU that defines `PYBIND11_MODULE`.

  :returns: Relative paths of the entry TU followed by the four families' sources.
  """

  return [
    "csrc/kernels/ops.cpp",
    *_required_svdquant_sources(),
    *_required_comm_sources(),
    *_required_attn_sources(),
    *_required_models_sources(),
  ]


def _missing_cuda_sources() -> list[str]:
  return [
    relative_path for relative_path in _required_cuda_sources()
    if not (ROOT_DIR / relative_path).exists()
  ]


def _get_nvcc_version(cuda_home: str | None) -> str:
  nvcc_path = path.join(cuda_home, "bin", "nvcc") if cuda_home else "nvcc"
  try:
    nvcc_output = subprocess.check_output([nvcc_path, "--version"]).decode()
  except (OSError, subprocess.CalledProcessError) as exc:
    raise RuntimeError(
      "A Cache-DiT CUDA extension build was requested, but `nvcc` was not found. "
      "Activate the CUDA toolchain inside `conda activate cdit` before building.") from exc

  match = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", nvcc_output)
  if match is None:
    raise RuntimeError(f"Unable to parse nvcc version from: {nvcc_output!r}")

  return match.group(2)


def _get_cuda_nvcc_threads() -> int:
  # One extension, so this thread count (`--threads=`) now applies to every family. The env name
  # predates the merge and is kept so existing build scripts keep working.
  raw_threads = os.getenv("CACHE_DIT_SVDQ_NVCC_THREADS", "8").strip()
  try:
    threads = int(raw_threads)
  except ValueError as exc:
    raise RuntimeError("CACHE_DIT_SVDQ_NVCC_THREADS must be a positive integer.") from exc

  if threads < 1:
    raise RuntimeError("CACHE_DIT_SVDQ_NVCC_THREADS must be a positive integer.")

  return threads


def _get_version_write_to() -> str | None:
  if VERSION_WRITE_TO_ENV not in os.environ:
    return path.join("src", "cache_dit", "_version.py")

  raw_value = os.getenv(VERSION_WRITE_TO_ENV, "").strip()
  return raw_value or None


def _get_cmdclass() -> dict[str, object]:
  cmdclass = dict(CMDCLASS)
  if _bdist_wheel is None:
    return cmdclass

  class CacheDiTBdistWheel(_bdist_wheel):

    def finalize_options(self):
      super().finalize_options()
      if IS_CU13_RELEASE:
        self.root_is_pure = False

    def get_tag(self):
      python_tag, abi_tag, platform_tag = super().get_tag()
      if IS_CU13_RELEASE:
        return python_tag, abi_tag, _get_release_platform_tag()
      return python_tag, abi_tag, platform_tag

  cmdclass["bdist_wheel"] = CacheDiTBdistWheel
  return cmdclass


def _validate_release_configuration() -> None:
  if not IS_CU13_RELEASE:
    return

  if not _should_build_cuda():
    raise RuntimeError("The 'cu13' release flavor requires CACHE_DIT_BUILD_CUDA=1 "
                       "(see docs/user_guide/INSTALL.md for the release commands).")

  if _get_cuda_build_mode() != "ALL":
    raise RuntimeError("The 'cu13' release flavor requires CACHE_DIT_CUDA_BUILD_MODE=ALL.")


def _resolve_sm_targets(scope: str, is_supported: Callable[[str], bool]) -> list[str]:
  """Resolve the CUDA SM targets one optional extension compiles for.

  Every branch applies `is_supported` to the candidates, so an extension never receives a
  `-gencode` flag its kernels cannot host: the predicate is the only gate, and there is no
  post-hoc validation that could be forgotten when a branch is edited. An explicit arch
  list stays authoritative -- it is never replaced by the default set -- so an extension
  whose targets all fail the predicate resolves to an empty list, and the caller decides
  between skipping it and failing.

  :param scope: Extension label used in the build messages, e.g. `"Comm"` or `"Attn"`.
  :param is_supported: Predicate deciding whether a normalized SM target can host the
    extension's kernels.
  :returns: The resolved SM targets, e.g. `["89", "90a", "120a"]`, or an empty list when
    no candidate qualifies.
  """

  assert packaging_version is not None
  assert CUDA_HOME is not None
  assert torch is not None

  nvcc_version = _get_nvcc_version(CUDA_HOME)
  support_sm90a = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.0")
  support_sm100 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")
  support_sm120 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")
  support_sm121 = packaging_version.parse(nvcc_version) >= packaging_version.parse("13.0")

  build_mode = _get_cuda_build_mode()

  # An empty value counts as unset: the release build passes
  # `CACHE_DIT_CUDA_ARCH_LIST=<>` and `TORCH_CUDA_ARCH_LIST=` on purpose, so that the
  # default architecture set is what gets compiled.
  explicit_arch_list = os.getenv("CACHE_DIT_CUDA_ARCH_LIST") or os.getenv("TORCH_CUDA_ARCH_LIST")
  if explicit_arch_list:
    arch_list_env = ("CACHE_DIT_CUDA_ARCH_LIST"
                     if os.getenv("CACHE_DIT_CUDA_ARCH_LIST") else "TORCH_CUDA_ARCH_LIST")
    candidates = [
      _normalize_sm_target(
        sm,
        support_sm120=support_sm120,
        support_sm121=support_sm121,
      ) for sm in _parse_arch_list(explicit_arch_list)
    ]
    source = f"from {arch_list_env}"
  elif build_mode == "FAST" and torch.cuda.is_available() and torch.cuda.device_count() > 0:
    candidates = []
    for device_index in range(torch.cuda.device_count()):
      capability = torch.cuda.get_device_capability(device_index)
      candidates.append(
        _normalize_sm_target(
          f"{capability[0]}{capability[1]}",
          support_sm120=support_sm120,
          support_sm121=support_sm121,
        ))
    source = "detected on this machine"
  else:
    candidates = _default_cuda_arch_targets(
      support_sm90a=support_sm90a,
      support_sm100=support_sm100,
      support_sm120=support_sm120,
    )
    source = ""

  sm_targets = list(dict.fromkeys(sm for sm in candidates if is_supported(sm)))
  skipped = list(dict.fromkeys(sm for sm in candidates if not is_supported(sm)))
  if skipped and source:
    _warn_skipped(f"[cache-dit] {scope}: skipping SM targets {source} that cannot host its "
                  f"kernels: {', '.join(skipped)}.")
  return sm_targets


def _require_min_cuda_for_cu13_release() -> None:
  if not IS_CU13_RELEASE:
    return

  assert packaging_version is not None
  nvcc_version = _get_nvcc_version(CUDA_HOME)
  if packaging_version.parse(nvcc_version) < packaging_version.parse("13.0"):
    raise RuntimeError("The 'cu13' release flavor requires CUDA 13.0+ for compilation, "
                       f"but nvcc reports {nvcc_version}.")


def _get_cuda_extension():
  """Build the single optional `cache_dit._C_ops` extension requested by `CACHE_DIT_BUILD_CUDA`.

  One extension links every kernel family, so there is one source list, one `-gencode` set and
  one header-dependency set. The SVDQuant sources need the spdlog and CUTLASS submodules, so
  those become prerequisites of the whole extension rather than of a single family.

  :returns: A single-element list holding the `cache_dit._C_ops` extension, or an empty list
    when the build does not request it.
  :raises SystemExit: If the build requested the extension but no resolved SM target can host
    it. There is only one extension, so skipping it would silently ship a CUDA-less wheel for a
    build that explicitly asked for CUDA.
  """

  if not _should_build_cuda():
    return []

  sm_targets = _resolve_sm_targets("CUDA", _is_supported_cuda_sm_target)
  if not sm_targets:
    reason = "no resolved SM target can host cache_dit._C_ops (>=sm89, excluding sm90)"
    message = (f"[cache-dit] {reason}. Set CACHE_DIT_CUDA_ARCH_LIST to a supported target, or "
               "unset CACHE_DIT_BUILD_CUDA for a pure-Python install.")
    if IS_CU13_RELEASE:
      raise RuntimeError("[cache-dit] The 'cu13' release flavor ships the CUDA extension, "
                         f"but {reason}.")
    raise SystemExit(message)

  _require_min_cuda_for_cu13_release()

  missing_sources = _missing_cuda_sources()
  if missing_sources:
    preview = ", ".join(missing_sources[:4])
    if len(missing_sources) > 4:
      preview += ", ..."
    raise RuntimeError(
      "`CACHE_DIT_BUILD_CUDA=1` was set, but the cache_dit._C_ops source list is incomplete. "
      f"Missing files under {ROOT_DIR}: {preview}")

  assert CUDAExtension is not None

  _ensure_spdlog_submodule()
  # The `-DBUILD_CACHE_DIT_*` macros are inert labels: no source references them. They are
  # kept so the compile command still states which families the module links.
  gcc_flags = [
    "-DENABLE_BF16=1",
    "-DBUILD_CACHE_DIT_SVDQUANT=1",
    "-DBUILD_CACHE_DIT_COMM=1",
    "-DBUILD_CACHE_DIT_ATTN=1",
    "-DBUILD_CACHE_DIT_MODELS=1",
    "-fvisibility=hidden",
    "-O3",
    "-std=c++20",
  ]
  msvc_flags = [
    "/DENABLE_BF16=1",
    "/DBUILD_CACHE_DIT_SVDQUANT=1",
    "/DBUILD_CACHE_DIT_COMM=1",
    "/DBUILD_CACHE_DIT_ATTN=1",
    "/DBUILD_CACHE_DIT_MODELS=1",
    "/O2",
    "/std:c++20",
    "/Zc:__cplusplus",
    "/FS",
  ]
  nvcc_flags = [
    "-DENABLE_BF16=1",
    "-DBUILD_CACHE_DIT_SVDQUANT=1",
    "-DBUILD_CACHE_DIT_COMM=1",
    "-DBUILD_CACHE_DIT_ATTN=1",
    "-DBUILD_CACHE_DIT_MODELS=1",
    "-O3",
    "-std=c++20",
    "-Xcudafe",
    "--diag_suppress=20208",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_HALF2_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    f"--threads={_get_cuda_nvcc_threads()}",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "--ptxas-options=--allow-expensive-optimizations=true",
  ]
  if os.getenv("CACHE_DIT_BUILD_WHEELS", "0") == "0":
    nvcc_flags.append("--generate-line-info")
  for sm_target in sm_targets:
    nvcc_flags += ["-gencode", f"arch=compute_{sm_target},code=sm_{sm_target}"]

  extension = CUDAExtension(
    name="cache_dit._C_ops",
    sources=_required_cuda_sources(),
    depends=[
      str(path) for path in (ROOT_DIR / "csrc" / "kernels").rglob("*")
      if path.suffix in {".h", ".cuh"}
    ],
    include_dirs=[
      str(ROOT_DIR / "csrc" / "kernels"),
      str(ROOT_DIR / "csrc" / "kernels" / "svdq"),
      str(ROOT_DIR / "csrc" / "kernels" / "comm"),
      str(ROOT_DIR / "csrc" / "kernels" / "attn"),
      str(ROOT_DIR / "csrc" / "kernels" / "models"),
      str(ROOT_DIR / "csrc" / "third_party" / "spdlog" / "include"),
      _get_cutlass_include_dir(),
    ],
    extra_compile_args={
      "gcc": gcc_flags,
      "msvc": msvc_flags,
      "nvcc": nvcc_flags,
      "nvcc_msvc": [
        "-Xcompiler",
        "/Zc:__cplusplus",
        "-Xcompiler",
        "/FS",
        "-Xcompiler",
        "/bigobj",
      ],
    },
  )

  return [extension]


def _get_extensions() -> tuple[list, dict[str, object]]:
  """Collect the CUDA extension requested by `CACHE_DIT_BUILD_CUDA`.

  :returns: The requested extension and the shared `build_ext` cmdclass mapping.
  :raises SystemExit: If the build requested the CUDA extension but no resolved SM target can
    host it; `_get_cuda_extension()` owns that check because skipping the only extension would
    produce a wheel that silently lacks every CUDA operator.
  """

  if not _should_build_cuda():
    return [], {}

  assert CacheDiTBuildExtension is not None
  return _get_cuda_extension(), {"build_ext": CacheDiTBuildExtension}


def is_git_directory(path="."):
  return (subprocess.call(
    ["git", "-C", path, "status"],
    stderr=subprocess.STDOUT,
    stdout=open(os.devnull, "w"),
  ) == 0)


def my_local_scheme(version):
  # The following is used to build release packages.
  # Users should never use it.
  local_version = os.getenv("CACHE_DIT_BUILD_LOCAL_VERSION")
  if local_version is None:
    return get_local_dirty_tag(version)
  return f"+{local_version}"


_validate_release_configuration()
EXT_MODULES, CMDCLASS = _get_extensions()
USE_SCM_VERSION = {
  "version_scheme": "python-simplified-semver",
  "local_scheme": my_local_scheme,
}
VERSION_WRITE_TO = _get_version_write_to()
if VERSION_WRITE_TO is not None:
  USE_SCM_VERSION["write_to"] = VERSION_WRITE_TO

setup(
  description=
  "Cache-DiT: A PyTorch-native Inference Engine with Cache, Parallelism, Quantization and CPU Offload for DiTs.",
  author="DefTruth, vipshop.com",
  use_scm_version=USE_SCM_VERSION,
  package_dir={"": "src"},
  packages=find_packages(where="src"),
  include_package_data=True,
  python_requires=">=3.11",
  ext_modules=EXT_MODULES,
  cmdclass=_get_cmdclass(),
)
