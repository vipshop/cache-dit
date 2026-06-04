#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

if [[ "${CACHE_DIT_CLEAN:-0}" == "1" ]]; then
  echo "[build_fast] CACHE_DIT_CLEAN=1 -> removing build/ and workspace egg-info"
  rm -rf build/ __pycache__
  find src -maxdepth 1 -type d -name '*.egg-info' -prune -exec rm -rf {} +
fi

if [[ $# -eq 0 ]]; then
  set -- wheel . --no-build-isolation --no-deps -w dist
fi

if [[ -z "${MAX_JOBS:-}" ]]; then
  export MAX_JOBS=32
fi

if [[ -n "${CACHE_DIT_BUILD_SVDQUANT:-}" && "${CACHE_DIT_BUILD_SVDQUANT}" != "0" ]]; then
  if ! command -v ccache >/dev/null 2>&1; then
    if [[ "${CACHE_DIT_REQUIRE_CCACHE:-0}" == "1" ]]; then
      echo "[build_fast] ccache is required for this build but was not found." >&2
      exit 1
    fi
    echo "[build_fast] ccache not found; nvcc caching disabled." >&2
  else
    REAL_CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
    if [[ ! -x "$REAL_CUDA_HOME/bin/nvcc" ]]; then
      echo "[build_fast] real nvcc not found under $REAL_CUDA_HOME/bin." >&2
      exit 1
    fi

    SHADOW_CUDA="$REPO_DIR/build/.ccache_cuda_home"
    mkdir -p "$SHADOW_CUDA/bin"
    for entry in "$REAL_CUDA_HOME"/*; do
      name="$(basename "$entry")"
      [[ "$name" == "bin" ]] && continue
      ln -sfn "$entry" "$SHADOW_CUDA/$name"
    done
    for entry in "$REAL_CUDA_HOME/bin"/*; do
      name="$(basename "$entry")"
      [[ "$name" == "nvcc" ]] && continue
      ln -sfn "$entry" "$SHADOW_CUDA/bin/$name"
    done

    chmod +x "$REPO_DIR/tools/nvcc"
    cp -f "$REPO_DIR/tools/nvcc" "$SHADOW_CUDA/bin/nvcc"
    export NVCC_REAL="$REAL_CUDA_HOME/bin/nvcc"
    export CUDA_HOME="$SHADOW_CUDA"
    export CUDA_PATH="$SHADOW_CUDA"
    export PATH="$SHADOW_CUDA/bin:$PATH"
    export CCACHE_MAXSIZE="${CCACHE_MAXSIZE:-20G}"
    echo "[build_fast] nvcc ccache shim active: CUDA_HOME=$CUDA_HOME (real nvcc=$NVCC_REAL)"
  fi
fi

echo "[build_fast] MAX_JOBS=$MAX_JOBS CACHE_DIT_RELEASE_FLAVOR=${CACHE_DIT_RELEASE_FLAVOR:-base}"
python -m pip "$@"
