#!/usr/bin/env bash
# Build the release wheels of cache-dit: the pure-Python wheel PyPI serves, plus
# the CUDA (SVDQuant) wheels of one CUDA channel.
#
# Both builds carry the same distribution name.  What separates the CUDA build is
# the local version of its wheel (+cu13) -- the same thing that names the index
# directory it is served from (whl/cu13/), one directory per CUDA toolkit.  A
# requirement therefore does not change with the toolkit, and pip can tell the two
# builds apart, including the case where the pure-Python wheel of the same release
# is already installed.
#
# Every compilation switch keeps the meaning setup.py documents; only the knobs a
# release has an opinion about are set here (the flavor, the build mode, the arch
# set, the version).  The workspace the build runs in is prepared by
# tools/release_workspace.py, so nothing lands in the development tree.
#
# Profiles:
#   mini (default) -- one conda env, one CUDA arch, and a version that always
#         carries .dev.  It exercises the whole build -> wheel -> retag -> index
#         -> install path in minutes; tools/publish_index.sh refuses its wheels.
#   full  -- py310..py314 and the setup.py default arch set (80,86,89,120a,121a):
#         the published set.  Needs a release version, refuses .devN, and refuses
#         sources that are not committed.
#
# Usage:
#   bash tools/build_releases.sh
#   CACHE_DIT_RELEASE_PROFILE=full CACHE_DIT_RELEASE_VERSION=1.5.3 bash tools/build_releases.sh
#   CACHE_DIT_RELEASE_ENVS="py310 py311 py312 py313 py314" bash tools/build_releases.sh
#
# Environment:
#   CACHE_DIT_RELEASE_PROFILE    mini | full                    (default mini)
#   CACHE_DIT_RELEASE_VERSION    version to build; empty -> from git
#   CACHE_DIT_RELEASE_ENVS       conda envs, space separated    (default per profile)
#   CACHE_DIT_RELEASE_DIST       where the wheels land          (default dist/)
#   CACHE_DIT_RELEASE_WORK_ROOT  build workspaces               (default .tmp/release_build)
#   CACHE_DIT_RELEASE_CHANNEL    CUDA channel: the local version suffix and the
#                                index directory                 (default cu13)
#   CACHE_DIT_RELEASE_ARCH       cu13 arch list; empty -> setup.py's own default
#                                set (mini defaults to a single arch)
#   CACHE_DIT_RELEASE_MINI_ARCH  arch of the mini profile       (default 120a)
#   CACHE_DIT_RELEASE_BASE_ENV   conda env of the pure wheel    (default cdit)
#   CACHE_DIT_RELEASE_JOBS       MAX_JOBS for the extension build
#   CACHE_DIT_RELEASE_ALLOW_UNCOMMITTED  1 -> accept uncommitted shipped inputs
#   CACHE_DIT_RELEASE_DRY_RUN    1 -> resolve and print, build nothing
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_NAME="${CACHE_DIT_RELEASE_NAME:-cache_dit}"
# The wheel filename carries the escaped distribution name (``-`` becomes ``_``),
# which is what every glob below looks for.
WHEEL_STEM="${PACKAGE_NAME//-/_}"
CHANNEL="${CACHE_DIT_RELEASE_CHANNEL:-cu13}"
PROFILE="${CACHE_DIT_RELEASE_PROFILE:-mini}"
DIST_DIR="${CACHE_DIT_RELEASE_DIST:-$REPO_DIR/dist}"
WORK_ROOT="${CACHE_DIT_RELEASE_WORK_ROOT:-$REPO_DIR/.tmp/release_build}"
BASE_WORKSPACE="$WORK_ROOT/base"
CU13_WORKSPACE="$WORK_ROOT/$CHANNEL"
BASE_ENV="${CACHE_DIT_RELEASE_BASE_ENV:-cdit}"
REQUESTED_VERSION="${CACHE_DIT_RELEASE_VERSION:-}"
# Both build functions rely on setuptools-scm owning the version: one of these left
# in the environment would send the version file back into the checkout, or rename
# the plain wheel with a local segment.
unset CACHE_DIT_VERSION_WRITE_TO CACHE_DIT_BUILD_LOCAL_VERSION
DRY_RUN="${CACHE_DIT_RELEASE_DRY_RUN:-0}"
ALLOW_UNCOMMITTED="${CACHE_DIT_RELEASE_ALLOW_UNCOMMITTED:-0}"

# Everything that decides what a wheel contains: the sources, the metadata, and the
# tools -- a change to tools/build_fast.sh, release_workspace.py or
# repair_wheel_platform_tag.py changes the artifact as much as a change to src/.
# This is what the guard of profile full looks at.
SHIPPED_INPUTS=(src csrc tools setup.py pyproject.toml setup.cfg MANIFEST.in README.md LICENSE)

fail() {
  echo "[build_releases] $1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

case "$PROFILE" in
  mini)
    DEFAULT_ENVS="py312"
    ;;
  full)
    DEFAULT_ENVS="py310 py311 py312 py313 py314"
    ;;
  *)
    fail "CACHE_DIT_RELEASE_PROFILE must be mini or full, got '$PROFILE'"
    ;;
esac

ENV_LIST_RAW="${CACHE_DIT_RELEASE_ENVS:-$DEFAULT_ENVS}"
read -r -a RELEASE_ENVS <<< "$ENV_LIST_RAW"

# An empty arch list means "whatever setup.py defaults to" (80,86,89,120a,121a on
# a CUDA 13 toolkit), which is the published set the full profile is for.  The
# mini profile narrows it to one arch: it is there to validate the pipeline.
release_arch() {
  if [[ -n "${CACHE_DIT_RELEASE_ARCH:-}" ]]; then
    echo "$CACHE_DIT_RELEASE_ARCH"
    return 0
  fi
  if [[ "$PROFILE" == "mini" ]]; then
    echo "${CACHE_DIT_RELEASE_MINI_ARCH:-120a}"
    return 0
  fi
  echo ""
}
ARCH="$(release_arch)"

expected_python() {
  case "$1" in
    py310) echo "3.10" ;;
    py311) echo "3.11" ;;
    py312) echo "3.12" ;;
    py313) echo "3.13" ;;
    py314) echo "3.14" ;;
    *) fail "unsupported release env '$1' (expected one of py310 py311 py312 py313 py314)" ;;
  esac
}

wheel_python_tag() {
  case "$1" in
    py310) echo "cp310" ;;
    py311) echo "cp311" ;;
    py312) echo "cp312" ;;
    py313) echo "cp313" ;;
    py314) echo "cp314" ;;
    *) fail "unsupported release env '$1'" ;;
  esac
}

check_python_env() {
  local env_name="$1" expected
  expected="$(expected_python "$env_name")"
  conda run -n "$env_name" python -c \
    "import sys; raise SystemExit(0 if f'{sys.version_info[0]}.{sys.version_info[1]}' == '$expected' else 1)" \
    || fail "conda env $env_name does not provide Python $expected"
}

check_release_env() {
  local env_name="$1"

  conda run -n "$env_name" python -c "import packaging, setuptools_scm, torch, wheel" \
    || fail "conda env $env_name is missing build dependencies (packaging/setuptools_scm/torch/wheel)"

  # The cu13 flavor is what decides whether an extension is compiled at all, so the
  # toolkit floor is checked for every profile: setup.py refuses a release flavor
  # below CUDA 13.0 anyway.  nvcc is looked up under CUDA_HOME rather than on PATH
  # (the toolkit is not on PATH on this machine, and torch resolves its own
  # CUDA_HOME the same way).
  local toolkit
  toolkit="$("$NVCC" --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p' | head -1)"
  [[ "$toolkit" =~ ^[0-9]+\.[0-9]+$ ]] \
    || fail "cannot parse the CUDA release out of $NVCC --version"
  [[ "${toolkit%%.*}" -ge 13 ]] \
    || fail "$NVCC is CUDA $toolkit; the $CHANNEL release flavor needs 13.0+"
}

resolve_version() {
  local env_name="$1" resolved
  # The repository root travels in the environment, not inside the -c program: a
  # path with a quote in it would end that program there.
  resolved="$(CACHE_DIT_REPO_ROOT="$REPO_DIR" conda run -n "$env_name" python -c \
    "import os, setuptools_scm; print(setuptools_scm.get_version( \
root=os.environ['CACHE_DIT_REPO_ROOT'], version_scheme='python-simplified-semver', \
local_scheme='no-local-version'))" \
    | tr -d '\r' | tail -n 1)"
  [[ -n "$resolved" ]] || fail "could not resolve a version from $REPO_DIR"
  echo "$resolved"
}

resolve_release_version() {
  local env_name="$1" version
  version="${REQUESTED_VERSION:-$(resolve_version "$env_name")}"
  [[ "$version" =~ ^[0-9]+(\.[0-9]+)*([._a-zA-Z0-9]+)?$ ]] \
    || fail "unexpected version '$version'"

  if [[ "$PROFILE" == "mini" ]]; then
    # A mini wheel exists to exercise the pipeline, and tools/publish_index.sh
    # refuses a .dev version: deriving the version keeps that property.  Past the
    # last tag setuptools-scm already yields 1.5.3.devN; on the tag itself it
    # yields the bare release, which .dev0 restores (and which still sorts below
    # that release, so it cannot be mistaken for it).
    [[ -z "$REQUESTED_VERSION" ]] || fail \
      "profile mini derives its own .dev version; CACHE_DIT_RELEASE_VERSION belongs to profile full"
    if [[ "$version" != *".dev"* ]]; then
      version="${version}.dev0"
      echo "[build_releases] profile mini forces a .dev version: $version" >&2
    fi
    echo "$version"
    return 0
  fi

  [[ "$version" != *".dev"* ]] || fail \
    "profile $PROFILE needs a release version, got '$version'. Tag the commit, or pass CACHE_DIT_RELEASE_VERSION=<x.y.z>"
  # A local segment belongs to the CUDA wheel alone: it is what tells pip this is
  # not the PyPI release, and PyPI refuses one anyway.
  [[ "$version" != *"+"* ]] || fail \
    "the pure-Python release carries a plain version; '$version' has a local segment"
  echo "$version"
}

check_shipped_inputs() {
  [[ "$PROFILE" == "full" ]] || return 0

  local uncommitted
  uncommitted="$(git -C "$REPO_DIR" status --porcelain -- "${SHIPPED_INPUTS[@]}")"
  if [[ -n "$uncommitted" && "$ALLOW_UNCOMMITTED" == "1" ]]; then
    echo "[build_releases] WARNING: building from uncommitted inputs; the wheels will not" >&2
    echo "[build_releases] WARNING: correspond to any commit:" >&2
    echo "$uncommitted" >&2
    return 0
  fi
  [[ -z "$uncommitted" ]] || fail \
    "profile $PROFILE builds from committed sources; commit these first (or set
CACHE_DIT_RELEASE_ALLOW_UNCOMMITTED=1 to accept artifacts that match no commit):
$uncommitted"
}

prepare_workspace() {
  local workspace="$1"
  echo "[build_releases] preparing $workspace"
  conda run -n "$BASE_ENV" python "$REPO_DIR/tools/release_workspace.py" \
    --source-root "$REPO_DIR" \
    --workspace "$workspace" \
    --package-name "$PACKAGE_NAME"
}

# Inspect what a wheel actually carries: the pure-Python wheel must hold no binary
# and the CUDA wheel must hold the extension; both must ship the same version in
# cache_dit/_version.py that the metadata advertises, or `cache_dit.__version__`
# would disagree with the wheel it came from.  The check lives in a script of its
# own rather than in a heredoc: `conda run` swallows the standard input of the
# program it starts, so anything fed through `python -` would be an empty program
# that exits successfully -- an assertion that passes without running.
inspect_wheel() {
  local wheel="$1" so_mode="$2" expect_version="$3"
  conda run -n "$BASE_ENV" python "$REPO_DIR/tools/inspect_release_wheel.py" \
    "$wheel" --so "$so_mode" --version "$expect_version" \
    || fail "wheel inspection failed: $wheel"
}

build_base_wheel() {
  echo "[build_releases] $BASE_ENV: building the pure-Python wheel"
  (
    # The version file is deliberately left to setuptools-scm (its default
    # location, which here is the workspace's copy of src/): it writes the module
    # the package imports __version__ from, in the shape that belongs to whatever
    # setuptools-scm version does the build.
    cd "$BASE_WORKSPACE"
    conda run --no-capture-output -n "$BASE_ENV" env \
      SETUPTOOLS_SCM_PRETEND_VERSION="$RELEASE_VERSION" \
      SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CACHE_DIT="$RELEASE_VERSION" \
      python -m pip wheel . --no-build-isolation --no-deps -w "$DIST_DIR"
  )
}

build_cu13_wheel() {
  local env_name="$1" python_tag wheels wheel raw retagged detected
  python_tag="$(wheel_python_tag "$env_name")"

  # A wheel for this (version, python) left by an earlier run -- or by the other
  # profile, whose filename is identical -- would satisfy the assertion below and
  # could be the one that ends up on the index.
  rm -f "$DIST_DIR"/"$WHEEL_STEM"-"$RELEASE_VERSION+$CHANNEL"-"$python_tag"-"$python_tag"-*.whl

  echo "[build_releases] $env_name: building the $CHANNEL wheel (arch: ${ARCH:-setup.py default})"
  (
    cd "$CU13_WORKSPACE"
    conda run --no-capture-output -n "$env_name" env \
      SETUPTOOLS_SCM_PRETEND_VERSION="$RELEASE_VERSION+$CHANNEL" \
      SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CACHE_DIT="$RELEASE_VERSION+$CHANNEL" \
      CACHE_DIT_RELEASE_FLAVOR=cu13 \
      CACHE_DIT_BUILD_SVDQUANT=1 \
      CACHE_DIT_BUILD_WHEELS=1 \
      CACHE_DIT_SVDQUANT_BUILD_MODE=ALL \
      CACHE_DIT_WHEEL_PLATFORM_TAG=linux_x86_64 \
      CACHE_DIT_CUDA_ARCH_LIST="$ARCH" \
      TORCH_CUDA_ARCH_LIST= \
      CACHE_DIT_REQUIRE_CCACHE=1 \
      CACHE_DIT_CLEAN=1 \
      ${CACHE_DIT_RELEASE_JOBS:+MAX_JOBS=$CACHE_DIT_RELEASE_JOBS} \
      bash tools/build_fast.sh wheel . --no-build-isolation --no-deps -w "$DIST_DIR"
  )

  shopt -s nullglob
  wheels=("$DIST_DIR"/"$WHEEL_STEM"-"$RELEASE_VERSION+$CHANNEL"-"$python_tag"-"$python_tag"-*.whl)
  shopt -u nullglob
  [[ ${#wheels[@]} -eq 1 ]] || fail \
    "$env_name: expected exactly one $CHANNEL wheel for $python_tag in $DIST_DIR, got ${#wheels[@]}"
  raw="${wheels[0]}"
  [[ "$(basename "$raw")" == "$WHEEL_STEM-$RELEASE_VERSION+$CHANNEL-$python_tag-$python_tag-"* ]] \
    || fail "$env_name: built $(basename "$raw") but expected version $RELEASE_VERSION+$CHANNEL"

  # Two things a wheel built in a conda environment cannot ship as-is: the platform
  # tag (the floor is derived from the GLIBC symbols the extension actually needs)
  # and the RPATH, which points at the build environment's lib directory.
  retagged="$(conda run -n "$BASE_ENV" python \
    "$REPO_DIR/tools/repair_wheel_platform_tag.py" --retag --set-rpath '$ORIGIN' "$raw")"
  detected="${retagged#*$'\t'}"
  echo "[build_releases] $env_name: platform tag $detected, RPATH \$ORIGIN"
}

verify_artifacts() {
  local env_name python_tag wheels base_wheels channel_wheels detected report

  shopt -s nullglob
  base_wheels=("$DIST_DIR"/"$WHEEL_STEM"-"$RELEASE_VERSION"-py3-none-any.whl)
  channel_wheels=("$DIST_DIR"/"$WHEEL_STEM"-"$RELEASE_VERSION+$CHANNEL-"*.whl)
  shopt -u nullglob

  [[ ${#base_wheels[@]} -eq 1 ]] || fail \
    "expected exactly one pure-Python wheel $WHEEL_STEM-$RELEASE_VERSION-py3-none-any.whl in $DIST_DIR"
  inspect_wheel "${base_wheels[0]}" none "$RELEASE_VERSION"

  # The published set is exactly one wheel per release env: anything else under this
  # version is a leftover of another build (a narrower CACHE_DIT_RELEASE_ENVS?) and
  # must not be folded into the publish glob.
  [[ ${#channel_wheels[@]} -eq ${#RELEASE_ENVS[@]} ]] || fail \
    "expected ${#RELEASE_ENVS[@]} $CHANNEL wheel(s) for $RELEASE_VERSION in $DIST_DIR, got ${#channel_wheels[@]}"

  for env_name in "${RELEASE_ENVS[@]}"; do
    python_tag="$(wheel_python_tag "$env_name")"
    wheels=("$DIST_DIR"/"$WHEEL_STEM"-"$RELEASE_VERSION+$CHANNEL"-"$python_tag"-"$python_tag"-*.whl)
    [[ ${#wheels[@]} -eq 1 ]] || fail \
      "expected exactly one $CHANNEL wheel for $python_tag in $DIST_DIR, got ${#wheels[@]}"
    inspect_wheel "${wheels[0]}" svdquant "$RELEASE_VERSION+$CHANNEL"

    report="$(conda run -n "$BASE_ENV" python \
      "$REPO_DIR/tools/repair_wheel_platform_tag.py" --print-rpath "${wheels[0]}")"
    detected="${report#*$'\t'}"
    detected="${detected%%$'\t'*}"
    detected_rpath="${report##*$'\t'}"
    [[ "$(basename "${wheels[0]}")" == *"-${detected}.whl" ]] || fail \
      "${wheels[0]} advertises a platform tag other than the detected $detected"
    [[ "$detected_rpath" == '$ORIGIN' ]] || fail \
      "${wheels[0]} carries RPATH '$detected_rpath'; a release wheel is relocatable (\$ORIGIN)"
  done
  shopt -u nullglob
}

echo "[build_releases] package=$PACKAGE_NAME profile=$PROFILE channel=$CHANNEL"
echo "[build_releases] envs=${RELEASE_ENVS[*]// /,} arch=${ARCH:-<setup.py default>}"
echo "[build_releases] dist=$DIST_DIR work_root=$WORK_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[build_releases] dry run: no workspace prepared, no wheel built"
  exit 0
fi

require_command conda
require_command ccache
require_command git

# torch resolves its toolkit through CUDA_HOME, falling back to the default install
# prefix; the release build pins both that root and the nvcc the checks use, so the
# checks and the compilation cannot end up on different toolkits.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
NVCC="$CUDA_HOME/bin/nvcc"
[[ -x "$NVCC" ]] || fail "nvcc not found at $NVCC; set CUDA_HOME to a CUDA 13 toolkit"

conda run -n "$BASE_ENV" python -c "import setuptools_scm, wheel" \
  || fail "conda env $BASE_ENV is missing base wheel build dependencies"

for env_name in "${RELEASE_ENVS[@]}"; do
  check_python_env "$env_name"
  check_release_env "$env_name"
done

RELEASE_VERSION="$(resolve_release_version "$BASE_ENV")"
echo "[build_releases] version=$RELEASE_VERSION"
if [[ "$PROFILE" == "mini" ]]; then
  echo "[build_releases] mini wheels validate the pipeline only; their version carries .dev"
  echo "[build_releases] and tools/publish_index.sh refuses it, so there is nothing to publish"
fi
check_shipped_inputs

mkdir -p "$DIST_DIR"
# Wheels of this version left behind by an earlier run are not part of this build:
# a rerun with a narrower CACHE_DIT_RELEASE_ENVS or a narrower arch set would
# otherwise fold them into the publish glob below.
stale=("$DIST_DIR"/"$WHEEL_STEM-$RELEASE_VERSION"*.whl)
if [[ -e "${stale[0]}" ]]; then
  echo "[build_releases] removing ${#stale[@]} wheel(s) of $RELEASE_VERSION from an earlier build"
  rm -f "${stale[@]}"
fi

prepare_workspace "$BASE_WORKSPACE"
prepare_workspace "$CU13_WORKSPACE"

build_base_wheel

for env_name in "${RELEASE_ENVS[@]}"; do
  build_cu13_wheel "$env_name"
done

verify_artifacts
echo "[build_releases] built ${#RELEASE_ENVS[@]} $CHANNEL wheel(s) and one pure-Python wheel in $DIST_DIR"
ls -lh "$DIST_DIR"/"$WHEEL_STEM"-"$RELEASE_VERSION"*.whl

if [[ "$PROFILE" == "mini" ]]; then
  echo "[build_releases] nothing to publish from a mini build"
  exit 0
fi
echo "[build_releases] upload the pure-Python wheel to PyPI:"
echo "  python -m twine upload \"$DIST_DIR\"/$WHEEL_STEM-$RELEASE_VERSION-py3-none-any.whl"
echo "[build_releases] publish the $CHANNEL channel with:"
echo "  bash tools/publish_index.sh --tag v$RELEASE_VERSION+$CHANNEL \\"
echo "    --wheels \"$DIST_DIR\"/$WHEEL_STEM-$RELEASE_VERSION+$CHANNEL-*.whl --site-repo <gh-pages worktree>"
