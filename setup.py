import importlib.util

if importlib.util.find_spec("setuptools_scm") is None:
  raise ImportError("setuptools-scm is not installed. Install it by `pip3 install setuptools-scm`")

import os
import re
import subprocess
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
SVDQUANT_BUILD_FLAG = "CACHE_DIT_BUILD_SVDQUANT"
COMM_BUILD_FLAG = "CACHE_DIT_BUILD_COMM"
ATTN_BUILD_FLAG = "CACHE_DIT_BUILD_ATTN"
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
# datacenter (100), and Blackwell consumer architecture-specific (120a). The
# SVDQuant extension filters out Hopper because its INT4 kernels require >=sm80
# and exclude sm90; its FP4 kernels still require sm120a or newer at runtime.
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
  :returns: The supported subset of `89`, `90a`, `100`, and `120a`.
  """

  targets = ["89"]
  if support_sm90a:
    targets.append("90a")
  if support_sm100:
    targets.append("100")
  if support_sm120:
    targets.append("120a")
  return targets


def _is_supported_svdquant_sm_target(sm: str) -> bool:
  normalized = sm.removesuffix("a")
  if not normalized.isdigit():
    return False

  capability = int(normalized)
  return capability >= 80 and capability != 90


def _is_supported_comm_sm_target(sm: str) -> bool:
  """Return whether a normalized SM target can host the communication kernels.

  The attention kernels share the same floor, so this predicate gates both `_C_comm`
  and `_C_attn`.

  :param sm: Normalized SM target such as `89`, `90`, or `120a`.
  :returns: `True` when the target is `>=sm89`. Unlike SVDQuant, Hopper (sm90) is
    supported.
  """

  normalized = sm.removesuffix("a")
  if not normalized.isdigit():
    return False

  return int(normalized) >= 89


def _env_flag(name: str) -> bool:
  return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


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


def _should_build_svdquant() -> bool:
  return _env_flag(SVDQUANT_BUILD_FLAG)


def _should_build_comm() -> bool:
  """Return whether the optional communication extension takes part in the build.

  A full `CACHE_DIT_BUILD_SVDQUANT=1` build ships both CUDA extensions, so the
  SVDQuant flag also enables the communication extension.

  :returns: `True` when either build flag requests the communication extension.
  """

  return _env_flag(COMM_BUILD_FLAG) or _should_build_svdquant()


def _should_build_attn() -> bool:
  """Return whether the optional attention extension takes part in the build.

  A full `CACHE_DIT_BUILD_SVDQUANT=1` build ships every CUDA extension, and a
  `CACHE_DIT_BUILD_COMM=1` build also builds `cache_dit._C_attn`, so
  `CACHE_DIT_BUILD_ATTN=1` is only needed to build the attention extension on its own.

  :returns: `True` when either build flag requests the attention extension.
  """

  return _env_flag(ATTN_BUILD_FLAG) or _should_build_comm()


if _should_build_svdquant() or _should_build_comm() or _should_build_attn():
  try:
    import torch
    from packaging import version as packaging_version
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
  except ImportError as exc:
    raise RuntimeError(
      "Building the optional Cache-DiT CUDA extensions requires `torch` and `packaging` "
      "to be installed in the active environment. Run `conda activate cdit` and "
      "initialize submodules with `git submodule update --init --recursive --force`, then install with "
      "`CACHE_DIT_BUILD_COMM=1 python3 -m pip install -e . --no-build-isolation` or "
      "`CACHE_DIT_BUILD_ATTN=1 python3 -m pip install -e . --no-build-isolation` or "
      "`CACHE_DIT_BUILD_SVDQUANT=1 python3 -m pip install -e . --no-build-isolation`.") from exc
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


def _svdquant_v2_launch_sources() -> list[str]:
  return [
    f"csrc/kernels/svdq/int4/gemm_w4a4_v2_launch_{dtype}_int4_stage{stage}.cu"
    for dtype in ("fp16", "bf16") for stage in (1, 2, 3)
  ]


def _required_svdquant_sources() -> list[str]:
  return [
    "csrc/kernels/svdq/pybind.cpp",
    "csrc/kernels/svdq/torch.cpp",
    "csrc/kernels/svdq/int4/gemm_w4a4_v2.cu",
    *_svdquant_v2_launch_sources(),
    # Build both INT4 and NVFP4 kernels
    "csrc/kernels/svdq/core/gemm_w4a4.cu",
    "csrc/kernels/svdq/int4/gemm_w4a4_launch_fp16_int4.cu",
    "csrc/kernels/svdq/int4/gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
    "csrc/kernels/svdq/fp4/gemm_w4a4_launch_fp16_fp4.cu",
    "csrc/kernels/svdq/int4/gemm_w4a4_launch_bf16_int4.cu",
    "csrc/kernels/svdq/fp4/gemm_w4a4_launch_bf16_fp4.cu",
  ]


def _missing_svdquant_sources() -> list[str]:
  return [
    relative_path for relative_path in _required_svdquant_sources()
    if not (ROOT_DIR / relative_path).exists()
  ]


def _required_comm_sources() -> list[str]:
  return [
    "csrc/kernels/comm/comm_ops.cpp",
    "csrc/kernels/comm/fp8_comm.cu",
  ]


def _missing_comm_sources() -> list[str]:
  return [
    relative_path for relative_path in _required_comm_sources()
    if not (ROOT_DIR / relative_path).exists()
  ]


def _required_attn_sources() -> list[str]:
  return [
    "csrc/kernels/attn/attn.cpp",
    "csrc/kernels/attn/merge_states.cu",
  ]


def _missing_attn_sources() -> list[str]:
  return [
    relative_path for relative_path in _required_attn_sources()
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


def _get_svdquant_nvcc_threads() -> int:
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

  if not _should_build_svdquant():
    raise RuntimeError("The 'cu13' release flavor requires CACHE_DIT_BUILD_SVDQUANT=1 "
                       "(see docs/user_guide/INSTALL.md for the release commands).")

  build_mode = os.getenv("CACHE_DIT_SVDQUANT_BUILD_MODE", "").strip().upper()
  if build_mode != "ALL":
    raise RuntimeError("The 'cu13' release flavor requires CACHE_DIT_SVDQUANT_BUILD_MODE=ALL.")


def _get_sm_targets() -> list[str]:
  assert packaging_version is not None
  assert CUDA_HOME is not None
  assert torch is not None

  nvcc_version = _get_nvcc_version(CUDA_HOME)
  support_sm90a = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.0")
  support_sm100 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")
  support_sm120 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")
  support_sm121 = packaging_version.parse(nvcc_version) >= packaging_version.parse("13.0")

  explicit_arch_list = os.getenv("CACHE_DIT_CUDA_ARCH_LIST") or os.getenv("TORCH_CUDA_ARCH_LIST")
  if explicit_arch_list:
    raw_targets = [
      _normalize_sm_target(
        sm,
        support_sm120=support_sm120,
        support_sm121=support_sm121,
      ) for sm in _parse_arch_list(explicit_arch_list)
    ]
    # Filter out SM targets that SVDQuant doesn't support (e.g. sm75 too old,
    # sm90 Hopper excluded).  TORCH_CUDA_ARCH_LIST often lists all archs that
    # PyTorch was built for, which may be a superset of what SVDQuant needs.
    sm_targets = [sm for sm in raw_targets if _is_supported_svdquant_sm_target(sm)]
    skipped = [sm for sm in raw_targets if sm not in sm_targets]
    if skipped:
      print(
        f"[cache-dit] SVDQuant: skipping unsupported SM targets from "
        f"{'CACHE_DIT_CUDA_ARCH_LIST' if os.getenv('CACHE_DIT_CUDA_ARCH_LIST') else 'TORCH_CUDA_ARCH_LIST'}: "
        f"{', '.join(skipped)}.  Use CACHE_DIT_CUDA_ARCH_LIST to override.")
  else:
    sm_targets = []

  build_mode = os.getenv("CACHE_DIT_SVDQUANT_BUILD_MODE", "ALL").upper()
  if build_mode not in {"FAST", "ALL"}:
    raise RuntimeError(
      f"Unsupported CACHE_DIT_SVDQUANT_BUILD_MODE={build_mode!r}. Expected 'FAST' or 'ALL'.")

  if not sm_targets and build_mode == "FAST" and torch.cuda.is_available(
  ) and torch.cuda.device_count() > 0:
    for device_index in range(torch.cuda.device_count()):
      capability = torch.cuda.get_device_capability(device_index)
      sm = _normalize_sm_target(
        f"{capability[0]}{capability[1]}",
        support_sm120=support_sm120,
        support_sm121=support_sm121,
      )
      if sm not in sm_targets:
        sm_targets.append(sm)
  elif not sm_targets:
    sm_targets = [
      sm for sm in _default_cuda_arch_targets(
        support_sm90a=support_sm90a,
        support_sm100=support_sm100,
        support_sm120=support_sm120,
      ) if _is_supported_svdquant_sm_target(sm)
    ]

  unsupported_targets = [sm for sm in sm_targets if not _is_supported_svdquant_sm_target(sm)]
  if unsupported_targets:
    raise RuntimeError("SVDQuant INT4 kernels require >=sm80 and exclude Hopper (sm90). "
                       "Unsupported CUDA SM targets: "
                       f"{', '.join(unsupported_targets)}")

  if not sm_targets:
    raise RuntimeError("No CUDA SM targets were resolved for the SVDQuant extension build.")

  return sm_targets


def _get_ops_sm_targets(scope: str) -> list[str]:
  """Resolve the CUDA SM targets used to compile an optional operator extension.

  The communication and attention kernels are the only requirement here, so the filter keeps every
  target `>=sm89` and, unlike SVDQuant, keeps Hopper (sm90). Targets that cannot host the kernels
  are dropped instead of raising, so that a build requested for other accelerators still succeeds.

  :param scope: Extension label used in the build messages, e.g. `"Comm"` or `"Attn"`.
  :returns: The resolved SM targets, e.g. `["89", "90", "120a"]`, or an empty list when none of the
    requested targets can host the kernels.
  """

  assert packaging_version is not None
  assert CUDA_HOME is not None
  assert torch is not None

  nvcc_version = _get_nvcc_version(CUDA_HOME)
  support_sm90a = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.0")
  support_sm100 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")
  support_sm120 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")
  support_sm121 = packaging_version.parse(nvcc_version) >= packaging_version.parse("13.0")

  explicit_arch_list = os.getenv("CACHE_DIT_CUDA_ARCH_LIST") or os.getenv("TORCH_CUDA_ARCH_LIST")
  sm_targets: list[str] = []
  if explicit_arch_list:
    arch_list_env = ("CACHE_DIT_CUDA_ARCH_LIST"
                     if os.getenv("CACHE_DIT_CUDA_ARCH_LIST") else "TORCH_CUDA_ARCH_LIST")
    raw_targets = [
      _normalize_sm_target(
        sm,
        support_sm120=support_sm120,
        support_sm121=support_sm121,
      ) for sm in _parse_arch_list(explicit_arch_list)
    ]
    sm_targets = [sm for sm in raw_targets if _is_supported_comm_sm_target(sm)]
    skipped = [sm for sm in raw_targets if sm not in sm_targets]
    if skipped:
      print(f"[cache-dit] {scope}: skipping SM targets below sm89 from {arch_list_env}: "
            f"{', '.join(skipped)}.")
  elif os.getenv("CACHE_DIT_SVDQUANT_BUILD_MODE", "ALL").upper() == "FAST" and \
      torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for device_index in range(torch.cuda.device_count()):
      capability = torch.cuda.get_device_capability(device_index)
      sm = _normalize_sm_target(
        f"{capability[0]}{capability[1]}",
        support_sm120=support_sm120,
        support_sm121=support_sm121,
      )
      if sm not in sm_targets:
        sm_targets.append(sm)
    skipped = [sm for sm in sm_targets if not _is_supported_comm_sm_target(sm)]
    if skipped:
      print(f"[cache-dit] {scope}: skipping detected SM targets below sm89: {', '.join(skipped)}.")
      sm_targets = [sm for sm in sm_targets if _is_supported_comm_sm_target(sm)]
  else:
    sm_targets = _default_cuda_arch_targets(
      support_sm90a=support_sm90a,
      support_sm100=support_sm100,
      support_sm120=support_sm120,
    )

  return sm_targets


def _require_min_cuda_for_cu13_release() -> None:
  if not IS_CU13_RELEASE:
    return

  assert packaging_version is not None
  nvcc_version = _get_nvcc_version(CUDA_HOME)
  if packaging_version.parse(nvcc_version) < packaging_version.parse("13.0"):
    raise RuntimeError("The 'cu13' release flavor requires CUDA 13.0+ for compilation, "
                       f"but nvcc reports {nvcc_version}.")


def _get_svdquant_extension():
  if not _should_build_svdquant():
    return []

  _ensure_spdlog_submodule()
  _require_min_cuda_for_cu13_release()

  missing_sources = _missing_svdquant_sources()
  if missing_sources:
    preview = ", ".join(missing_sources[:4])
    if len(missing_sources) > 4:
      preview += ", ..."
    raise RuntimeError(
      "`CACHE_DIT_BUILD_SVDQUANT=1` was set, but the migrated native source tree is incomplete. "
      f"Missing files under {ROOT_DIR}: {preview}")

  assert CUDAExtension is not None

  sm_targets = _get_sm_targets()
  gcc_flags = [
    "-DENABLE_BF16=1",
    "-DBUILD_CACHE_DIT_SVDQUANT=1",
    "-fvisibility=hidden",
    "-O3",
    "-std=c++20",
  ]
  msvc_flags = [
    "/DENABLE_BF16=1",
    "/DBUILD_CACHE_DIT_SVDQUANT=1",
    "/O2",
    "/std:c++20",
    "/Zc:__cplusplus",
    "/FS",
  ]
  nvcc_flags = [
    "-DENABLE_BF16=1",
    "-DBUILD_CACHE_DIT_SVDQUANT=1",
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
    f"--threads={_get_svdquant_nvcc_threads()}",
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
    name="cache_dit._C_svdquant",
    sources=_required_svdquant_sources(),
    include_dirs=[
      str(ROOT_DIR / "csrc" / "kernels"),
      str(ROOT_DIR / "csrc" / "kernels" / "svdq"),
      str(ROOT_DIR / "csrc" / "third_party" / "spdlog" / "include"),
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


def _get_comm_extension():
  """Build the optional communication extension requested by the build flags.

  :returns: A single-element list holding the `cache_dit._C_comm` extension, or an empty
    list when the extension is not requested or cannot target this build.
  """

  if not _should_build_comm():
    return []

  _require_min_cuda_for_cu13_release()

  missing_sources = _missing_comm_sources()
  if missing_sources:
    preview = ", ".join(missing_sources[:4])
    if len(missing_sources) > 4:
      preview += ", ..."
    raise RuntimeError(
      "The Cache-DiT communication extension was requested, but its native source tree is "
      f"incomplete. Missing files under {ROOT_DIR}: {preview}")

  sm_targets = _get_ops_sm_targets("Comm")
  if not sm_targets:
    skip_reason = ("[cache-dit] Comm: no SM target >=sm89 was resolved, so the optional "
                   "`cache_dit._C_comm` extension is skipped. Set CACHE_DIT_CUDA_ARCH_LIST "
                   "to an explicit target list to build it.")
    if IS_CU13_RELEASE:
      raise RuntimeError(skip_reason)
    print(skip_reason)
    return []

  assert CUDAExtension is not None

  gcc_flags = [
    "-DBUILD_CACHE_DIT_COMM=1",
    "-fvisibility=hidden",
    "-O3",
    "-std=c++20",
  ]
  msvc_flags = [
    "/DBUILD_CACHE_DIT_COMM=1",
    "/O2",
    "/std:c++20",
    "/Zc:__cplusplus",
    "/FS",
  ]
  # The communication kernels only need relaxed constexpr and lambda evaluation, so the
  # SVDQuant-only fast-math and spdlog settings are intentionally left out.
  nvcc_flags = [
    "-DBUILD_CACHE_DIT_COMM=1",
    "-O3",
    "-std=c++20",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_HALF2_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
  ]
  if os.getenv("CACHE_DIT_BUILD_WHEELS", "0") == "0":
    nvcc_flags.append("--generate-line-info")
  for sm_target in sm_targets:
    nvcc_flags += ["-gencode", f"arch=compute_{sm_target},code=sm_{sm_target}"]

  extension = CUDAExtension(
    name="cache_dit._C_comm",
    sources=_required_comm_sources(),
    include_dirs=[
      str(ROOT_DIR / "csrc" / "kernels"),
      str(ROOT_DIR / "csrc" / "kernels" / "comm"),
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


def _get_attn_extension():
  """Build the optional attention extension requested by the build flags.

  :returns: A single-element list holding the `cache_dit._C_attn` extension, or an empty
    list when the extension is not requested or cannot target this build.
  """

  if not _should_build_attn():
    return []

  _require_min_cuda_for_cu13_release()

  missing_sources = _missing_attn_sources()
  if missing_sources:
    preview = ", ".join(missing_sources[:4])
    if len(missing_sources) > 4:
      preview += ", ..."
    raise RuntimeError(
      "The Cache-DiT attention extension was requested, but its native source tree is "
      f"incomplete. Missing files under {ROOT_DIR}: {preview}")

  sm_targets = _get_ops_sm_targets("Attn")
  if not sm_targets:
    skip_reason = ("[cache-dit] Attn: no SM target >=sm89 was resolved, so the optional "
                   "`cache_dit._C_attn` extension is skipped. Set CACHE_DIT_CUDA_ARCH_LIST "
                   "to an explicit target list to build it.")
    # An explicit CACHE_DIT_BUILD_ATTN also raises, because nothing else in the build
    # ships these operators and a silent skip would only surface at the first Ring merge.
    if IS_CU13_RELEASE or _env_flag(ATTN_BUILD_FLAG):
      raise RuntimeError(skip_reason)
    print(skip_reason)
    return []

  assert CUDAExtension is not None

  gcc_flags = [
    "-fvisibility=hidden",
    "-O3",
    "-std=c++20",
  ]
  msvc_flags = [
    "/O2",
    "/std:c++20",
    "/Zc:__cplusplus",
    "/FS",
  ]
  # The attention kernels only need relaxed constexpr and lambda evaluation, so the
  # SVDQuant-only fast-math and spdlog settings are intentionally left out.
  nvcc_flags = [
    "-O3",
    "-std=c++20",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_HALF2_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
  ]
  if os.getenv("CACHE_DIT_BUILD_WHEELS", "0") == "0":
    nvcc_flags.append("--generate-line-info")
  for sm_target in sm_targets:
    nvcc_flags += ["-gencode", f"arch=compute_{sm_target},code=sm_{sm_target}"]

  extension = CUDAExtension(
    name="cache_dit._C_attn",
    sources=_required_attn_sources(),
    include_dirs=[
      str(ROOT_DIR / "csrc" / "kernels"),
      str(ROOT_DIR / "csrc" / "kernels" / "attn"),
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
  """Collect every CUDA extension requested by the build flags.

  :returns: The requested extensions and the shared `build_ext` cmdclass mapping.
  """

  ext_modules = [*_get_svdquant_extension(), *_get_comm_extension(), *_get_attn_extension()]
  if not ext_modules:
    return [], {}

  assert CacheDiTBuildExtension is not None
  return ext_modules, {"build_ext": CacheDiTBuildExtension}


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
  "Cache-DiT: Cache-DiT: A PyTorch-native Inference Engine with Cache, Parallelism, Quantization and CPU Offload for DiTs.",
  author="DefTruth, vipshop.com",
  use_scm_version=USE_SCM_VERSION,
  package_dir={"": "src"},
  packages=find_packages(where="src"),
  include_package_data=True,
  python_requires=">=3.10",
  ext_modules=EXT_MODULES,
  cmdclass=_get_cmdclass(),
)
