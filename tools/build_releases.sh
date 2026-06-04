#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$REPO_DIR/dist"
WORK_ROOT="$REPO_DIR/.tmp/release_build"
BASE_WORKSPACE="$WORK_ROOT/base"
CU13_WORKSPACE="$WORK_ROOT/cu13"
BASE_ENV="${CACHE_DIT_BASE_ENV:-cdit}"
ENV_LIST_RAW="${CACHE_DIT_RELEASE_ENVS:-py310 py311 py312 py313 py314}"
RELEASE_VERSION=""

read -r -a RELEASE_ENVS <<< "$ENV_LIST_RAW"

fail() {
  echo "[build_releases] $1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

prepare_release_workspace() {
  local workspace="$1"
  local package_name="$2"

  python "$REPO_DIR/tools/release_workspace.py" \
    --source-root "$REPO_DIR" \
    --workspace "$workspace" \
    --package-name "$package_name"
}

resolve_release_version() {
  conda run -n "$BASE_ENV" python - <<PY
import os

import setuptools_scm
from setuptools_scm.version import get_local_dirty_tag


def my_local_scheme(version):
  local_version = os.getenv("CACHE_DIT_BUILD_LOCAL_VERSION")
  if local_version is None:
    return get_local_dirty_tag(version)
  return f"+{local_version}"


print(
  setuptools_scm.get_version(
    root=r"$REPO_DIR",
    version_scheme="python-simplified-semver",
    local_scheme=my_local_scheme,
  ))
PY
}

expected_python_for_env() {
  case "$1" in
    py310) echo "3.10" ;;
    py311) echo "3.11" ;;
    py312) echo "3.12" ;;
    py313) echo "3.13" ;;
    py314) echo "3.14" ;;
    *) fail "unsupported release env: $1" ;;
  esac
}

check_python_env() {
  local env_name="$1"
  local expected="$2"

  conda run -n "$env_name" python -c "import sys; actual=f'{sys.version_info[0]}.{sys.version_info[1]}'; raise SystemExit(0 if actual == '$expected' else 1)" \
    || fail "conda env $env_name does not provide Python $expected"
}

check_release_env() {
  local env_name="$1"

  conda run -n "$env_name" python -c "import packaging, setuptools_scm, torch, wheel" \
    || fail "conda env $env_name is missing build dependencies"

  conda run -n "$env_name" python - <<'PY' || fail "conda env ${env_name} does not satisfy CUDA 13.0+"
import re
import subprocess

from packaging import version

output = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release (\d+\.\d+), V(\d+\.\d+\.\d+)", output)
if match is None:
  raise SystemExit(1)
if version.parse(match.group(2)) < version.parse("13.0"):
  raise SystemExit(1)
PY
}

build_base_wheel() {
  echo "[build_releases] Building base wheel in $BASE_ENV"
  (
    cd "$BASE_WORKSPACE"
    conda run -n "$BASE_ENV" env \
      SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CACHE_DIT="$RELEASE_VERSION" \
      CACHE_DIT_VERSION_WRITE_TO= \
      python -m pip wheel . --no-build-isolation --no-deps -w "$DIST_DIR"
  )
}

build_cu13_wheel() {
  local env_name="$1"

  echo "[build_releases] Building cache-dit-cu13 wheel in $env_name with CACHE_DIT_SVDQUANT_BUILD_MODE=ALL and cleared CUDA arch overrides"
  (
    cd "$CU13_WORKSPACE"
    conda run -n "$env_name" env \
      SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CACHE_DIT_CU13="$RELEASE_VERSION" \
      CACHE_DIT_RELEASE_FLAVOR=cu13 \
      CACHE_DIT_BUILD_SVDQUANT=1 \
      CACHE_DIT_BUILD_WHEELS=1 \
      CACHE_DIT_SVDQUANT_BUILD_MODE=ALL \
      CACHE_DIT_CUDA_ARCH_LIST= \
      TORCH_CUDA_ARCH_LIST= \
      CACHE_DIT_REQUIRE_CCACHE=1 \
      CACHE_DIT_CLEAN=1 \
      CACHE_DIT_VERSION_WRITE_TO= \
      MAX_JOBS=32 \
      bash tools/build_fast.sh wheel . --no-build-isolation --no-deps -w "$DIST_DIR"
  )
}

verify_artifacts() {
  shopt -s nullglob
  local base_wheels=("$DIST_DIR"/cache_dit-*-py3-none-any.whl)
  [[ ${#base_wheels[@]} -eq 1 ]] || fail "expected exactly one base wheel in $DIST_DIR"

  for tag in cp310 cp311 cp312 cp313 cp314; do
    local wheels=("$DIST_DIR"/cache_dit_cu13-*-${tag}-${tag}-manylinux_2_34_x86_64.whl)
    [[ ${#wheels[@]} -eq 1 ]] || fail "expected exactly one cu13 wheel for ${tag} in $DIST_DIR"
  done
  shopt -u nullglob
}

require_command conda
require_command ccache
prepare_release_workspace "$BASE_WORKSPACE" "cache_dit"
prepare_release_workspace "$CU13_WORKSPACE" "cache-dit-cu13"

rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

conda run -n "$BASE_ENV" python -c "import setuptools_scm, wheel" \
  || fail "conda env $BASE_ENV is missing base wheel build dependencies"

RELEASE_VERSION="$(resolve_release_version)"
[[ -n "$RELEASE_VERSION" ]] || fail "failed to resolve release version from $REPO_DIR"
echo "[build_releases] Resolved release version: $RELEASE_VERSION"

for env_name in "${RELEASE_ENVS[@]}"; do
  check_python_env "$env_name" "$(expected_python_for_env "$env_name")"
  check_release_env "$env_name"
done

build_base_wheel

for env_name in "${RELEASE_ENVS[@]}"; do
  build_cu13_wheel "$env_name"
done

verify_artifacts
echo "[build_releases] all release wheels are available in $DIST_DIR"
