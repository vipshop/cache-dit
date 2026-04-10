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

PACKAGE_NAME = "cache-dit"
ROOT_DIR = Path(__file__).resolve().parent
SVDQUANT_BUILD_FLAG = "CACHE_DIT_BUILD_SVDQUANT"
SPDLOG_SUBMODULE_PATH = ROOT_DIR / "csrc" / "third_party" / "spdlog"
SPDLOG_HEADER_PATH = SPDLOG_SUBMODULE_PATH / "include" / "spdlog" / "spdlog.h"
CUDA_ARCH_ALIASES = {
  "maxwell": "50",
  "pascal": "60",
  "volta": "70",
  "turing": "75",
  "ampere": "80",
  "ada": "89",
  "hopper": "90",
}


def _env_flag(name: str) -> bool:
  return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _should_build_svdquant() -> bool:
  return _env_flag(SVDQUANT_BUILD_FLAG)


if _should_build_svdquant():
  try:
    import torch
    from packaging import version as packaging_version
    from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension
  except ImportError as exc:
    raise RuntimeError(
      "Building the optional SVDQuant extension requires `torch` and `packaging` "
      "to be installed in the active environment. Run `conda activate cdit` and "
      "initialize submodules with `git submodule update --init --recursive --force`, then install with "
      "`CACHE_DIT_BUILD_SVDQUANT=1 python3 -m pip install -e . --no-build-isolation`.") from exc
else:
  torch = None
  packaging_version = None
  CUDA_HOME = None
  BuildExtension = None
  CUDAExtension = None


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
    f"csrc/kernels/svdq/gemm_w4a4_v2_launch_{dtype}_int4_stage{stage}.cu"
    for dtype in ("fp16", "bf16") for stage in (1, 2, 3)
  ]


def _required_svdquant_sources() -> list[str]:
  return [
    "csrc/kernels/svdq/pybind.cpp",
    "csrc/kernels/svdq/torch.cpp",
    "csrc/kernels/svdq/gemm_w4a4_v2.cu",
    *_svdquant_v2_launch_sources(),
    "csrc/kernels/svdq/gemm_w4a4.cu",
    "csrc/kernels/svdq/gemm_w4a4_launch_fp16_int4.cu",
    "csrc/kernels/svdq/gemm_w4a4_launch_fp16_int4_fasteri2f.cu",
    "csrc/kernels/svdq/gemm_w4a4_launch_fp16_fp4.cu",
    "csrc/kernels/svdq/gemm_w4a4_launch_bf16_int4.cu",
    "csrc/kernels/svdq/gemm_w4a4_launch_bf16_fp4.cu",
  ]


def _missing_svdquant_sources() -> list[str]:
  return [
    relative_path for relative_path in _required_svdquant_sources()
    if not (ROOT_DIR / relative_path).exists()
  ]


def _get_nvcc_version(cuda_home: str | None) -> str:
  nvcc_path = path.join(cuda_home, "bin", "nvcc") if cuda_home else "nvcc"
  try:
    nvcc_output = subprocess.check_output([nvcc_path, "--version"]).decode()
  except (OSError, subprocess.CalledProcessError) as exc:
    raise RuntimeError(
      "`CACHE_DIT_BUILD_SVDQUANT=1` was set, but `nvcc` was not found. "
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


def _get_sm_targets() -> list[str]:
  assert packaging_version is not None
  assert CUDA_HOME is not None
  assert torch is not None

  explicit_arch_list = os.getenv("CACHE_DIT_CUDA_ARCH_LIST") or os.getenv("TORCH_CUDA_ARCH_LIST")
  if explicit_arch_list:
    return _parse_arch_list(explicit_arch_list)

  nvcc_version = _get_nvcc_version(CUDA_HOME)
  support_sm120 = packaging_version.parse(nvcc_version) >= packaging_version.parse("12.8")
  support_sm121 = packaging_version.parse(nvcc_version) >= packaging_version.parse("13.0")

  install_mode = os.getenv("CACHE_DIT_INSTALL_MODE", "FAST").upper()
  if install_mode not in {"FAST", "ALL"}:
    raise RuntimeError(
      f"Unsupported CACHE_DIT_INSTALL_MODE={install_mode!r}. Expected 'FAST' or 'ALL'.")

  sm_targets: list[str] = []
  if install_mode == "FAST" and torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for device_index in range(torch.cuda.device_count()):
      capability = torch.cuda.get_device_capability(device_index)
      sm = f"{capability[0]}{capability[1]}"
      if sm == "120" and support_sm120:
        sm = "120a"
      if sm == "121" and support_sm121:
        sm = "121a"
      if sm not in sm_targets:
        sm_targets.append(sm)
  else:
    sm_targets = ["75", "80", "86", "89"]
    if support_sm120:
      sm_targets.append("120a")
    if support_sm121:
      sm_targets.append("121a")

  if not sm_targets:
    raise RuntimeError("No CUDA SM targets were resolved for the SVDQuant extension build.")

  return sm_targets


def _get_svdquant_extension():
  if not _should_build_svdquant():
    return [], {}

  _ensure_spdlog_submodule()

  missing_sources = _missing_svdquant_sources()
  if missing_sources:
    preview = ", ".join(missing_sources[:4])
    if len(missing_sources) > 4:
      preview += ", ..."
    raise RuntimeError(
      "`CACHE_DIT_BUILD_SVDQUANT=1` was set, but the migrated native source tree is incomplete. "
      f"Missing files under {ROOT_DIR}: {preview}")

  assert BuildExtension is not None
  assert CUDAExtension is not None

  class CacheDiTBuildExtension(BuildExtension):

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

  return [extension], {"build_ext": CacheDiTBuildExtension}


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


EXT_MODULES, CMDCLASS = _get_svdquant_extension()

setup(
  name=PACKAGE_NAME,
  description=
  "Cache-DiT: Cache-DiT: A PyTorch-native Inference Engine with Cache, Parallelism and Quantization for Diffusion Transformers.",
  author="DefTruth, vipshop.com",
  use_scm_version={
    "write_to": path.join("src", "cache_dit", "_version.py"),
    "local_scheme": my_local_scheme,
  },
  package_dir={"": "src"},
  packages=find_packages(where="src"),
  include_package_data=True,
  python_requires=">=3.12",
  ext_modules=EXT_MODULES,
  cmdclass=CMDCLASS,
  entry_points={
    "console_scripts": [
      "cache-dit-metrics = cache_dit.metrics:main",  # metric entrypoint
      "cache-dit-generate = cache_dit.generate:main",  # example entrypoint
    ],
  },
)
