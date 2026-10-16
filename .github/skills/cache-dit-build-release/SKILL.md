---
name: cache-dit-build-release
description: "cache-dit 发布全流程（给定一个版本号即可执行完）：full profile 一次构建 PyPI 纯 Python wheel 与带 SVDQuant CUDA 扩展的 +cu13 wheel（cp310–cp314，sm 80/86/89/120a/121a，manylinux 平台标签与 $ORIGIN RPATH 归一）、gh-pages 上的 PEP 503 静态索引发布、GitHub Release 资产上传、PyPI 上传，以及发布后验收与安装语义。Use when cutting a cache-dit release, building or publishing the CUDA-extension wheels under whl/cu13, regenerating the wheel index on gh-pages, uploading the pure-Python wheel to PyPI, or verifying an install from the index."
user-invocable: true
---

# cache-dit 发布（build & release）

一次发布 = 同一个版本号下产出两件东西：**PyPI 的纯 Python 包**与**自建索引上的 SVDQuant CUDA 扩展包**。
两者同名同包（`cache_dit/`），靠 wheel 的 local 版本区分：CUDA 包是 `1.5.3+cu13`，纯包是 `1.5.3`。
除版本号 `V`（如 `1.5.4`）外无需其它输入；通道 `CHANNEL` 默认 `cu13`，除新增 CUDA toolkit 外不要改。
下文所有路径相对 **cache-dit 仓库根目录**（本机 `/workspace/dev/vipshop/cache-dit`）。

## 产物清单（以 `${V}` 为例）

| 产物 | 名称 | 去向 |
|---|---|---|
| CUDA wheel ×5 | `cache_dit-${V}+cu13-cp3{10..14}-cp3{10..14}-manylinux_2_38_x86_64.whl`（各 ~32 MB） | Release `v${V}+cu13` 资产 + gh-pages 索引 `whl/cu13/` |
| 纯 Python wheel ×1 | `cache_dit-${V}-py3-none-any.whl`（~0.5 MB） | PyPI（Release `v${V}` 挂 notes；资产可附可不附，见第 4 步） |
| 索引页 | `whl/cu13/cache-dit/index.html`（PEP 503，每个锚点带 sha256） | 项目 `gh-pages` 分支根 → `vipshop.github.io/cache-dit/` |

安装口径（发布后用户侧）：

```bash
pip install -U cache-dit                                                     # 默认：PyPI 纯 Python 包
pip install cache-dit --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/   # CUDA 扩展包，如 1.5.3+cu13
```

注意与 ffpa-attn 的差异：cache-dit 的自建索引只放 `cache-dit` 一个项目、不带依赖，因此必须用
`--extra-index-url`（PyPI 保持依赖来源），且要先装 CUDA 13 构建的 torch，详见
`references/install-and-index.md`。

## 执行顺序（勿跳步）

### 0. 预检

```bash
git status --porcelain -- src csrc tools setup.py pyproject.toml setup.cfg MANIFEST.in README.md LICENSE
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version | tail -2     # 需 CUDA 13.x（cu13 flavor 硬性要求）
for e in py310 py311 py312 py313 py314; do \
  conda run -n "$e" python -c \
  "import sys, packaging, setuptools_scm, torch, wheel; print(sys.version.split()[0], torch.__version__)"; done
command -v ccache patchelf objdump gh twine
gh auth status
gh release view "v${V}+cu13" --repo vipshop/cache-dit            # 期望 not found
```

- 第一条输出必须为空：full 档只从已提交源构建。确有需要可 `CACHE_DIT_RELEASE_ALLOW_UNCOMMITTED=1` 绕过（会告警，产物不对应任何 commit）。
- PyPI 版本不可覆盖：确认 `cache-dit` 上没有 V（`curl -s https://pypi.org/pypi/cache-dit/json`）。
- py310–py314 提供发布用的 CPython 集合，缺哪个就少一个 wheel；发布集合以实际 env 为准。本机 5 个 env 均已存在（`conda env list`）。
- gh-pages worktree：仓库常备 `.tmp/gh-pages`（`git worktree list` 可见），直接复用；不存在则 `git worktree add .tmp/gh-pages gh-pages`。

### 1. dry-run（可选，首次发布或改过发布脚本时）

```bash
CACHE_DIT_RELEASE_PROFILE=full CACHE_DIT_RELEASE_VERSION=$V CACHE_DIT_RELEASE_DRY_RUN=1 \
  bash tools/build_releases.sh
```

### 2. 构建全部产物（full 档：纯 wheel + CUDA wheel 一次出齐）

```bash
CACHE_DIT_RELEASE_PROFILE=full CACHE_DIT_RELEASE_VERSION=$V CACHE_DIT_RELEASE_CHANNEL=cu13 \
  CACHE_DIT_RELEASE_ENVS="py310 py311 py312 py313 py314" \
  bash tools/build_releases.sh
```

- 与 ffpa-attn 不同：cache-dit 没有单独的 pure profile。full 档在同一个 base env（默认 `cdit`）里先出纯 wheel，再逐 env 出 cu13 wheel，最后统一校验（纯包无 `.so`、CUDA 包含 SVDQuant 扩展、平台标签与 `$ORIGIN` RPATH）。
- 编译 arch 留空即 setup.py 默认集（CUDA 13 下为 `80,86,89,120a,121a`），这就是发布集；不要用 `CACHE_DIT_RELEASE_ARCH` 收窄，也不要包含 sm90（SVDQuant INT4 kernel 排除 Hopper）。
- 重发同版本：无 incremental 开关，但 ccache 命中后重编很快；workspace 在 `.tmp/release_build/{base,cu13}`，删掉即可全新重来。
- 只做回归验证、不碰已发布件：`CACHE_DIT_RELEASE_DIST=.tmp/<task>/dist`。
- 编译耗时较长（ccache 无命中时更久）：发出命令后等它结束，不要轮询。

### 3. 发布 CUDA 通道（Release 资产 + gh-pages 索引）

```bash
git -C .tmp/gh-pages pull --rebase          # 复用已有 worktree 时先同步
bash tools/publish_index.sh --tag "v${V}+cu13" \
  --wheels "dist/cache_dit-${V}+cu13-"*.whl --site-repo .tmp/gh-pages
```

一次调用做完四件事：`gh release create v${V}+cu13`（`--latest=false`、`--target` 当前 HEAD、notes 写 `--extra-index-url` 安装命令）→ 上传全部 wheel 资产（幂等：size + digest/索引 sha256 一致才跳过）→ 按 merge 语义重生成索引（历史版本保留）→ commit 并 push `gh-pages`。

- tag 规范 `v<版本>+<通道>`，与 wheel 的 local 版本同构；不一致会直接报错。
- 同名资产已存在且内容不同：脚本拒绝，需 `CACHE_DIT_FORCE_UPLOAD=1` 显式替换（或换版本号）。
- push 被拒时按打印的提示处理：`git -C .tmp/gh-pages pull --rebase && git -C .tmp/gh-pages push`。

### 4. 上传纯包（PyPI + 普通 Release）

```bash
python -m twine upload "dist/cache_dit-${V}-py3-none-any.whl"
gh release create "v${V}" --repo vipshop/cache-dit --target <发布提交> \
  --notes-file <notes.md> "dist/cache_dit-${V}-py3-none-any.whl"
```

- release notes 用 `release-notes-generator` skill 从 git 历史生成，与历史 release 的 Summary 形态一致。
- `--target` 不传时 gh 用默认分支（main）打 tag；发布提交不在 main 上时必须显式给。
- twine 403 表示 token 没有该项目权限。
- 历史实践：`v1.5.3` 的 release 只挂了 notes、**未附 wheel 资产**（纯包以 PyPI 为准）。附资产可选；若附，确保与 PyPI 上的是同一个文件。
- 纯包与 CUDA 通道最好来自**同一个发布提交**。历史现状是补发形态：`v1.5.3+cu13` 指向 `9109a46`（RPATH 修复）、`v1.5.3` 指向 `b7f8276`，两者不是同一修订。

### 5. 发布后验收（必做）

```bash
mkdir -p .tmp/cache-dit-build-release
python3 -m http.server -d .tmp/gh-pages 8765 &          # 本地服务索引工作树
pip download "cache-dit==${V}+cu13" --no-deps \
  -d .tmp/cache-dit-build-release --index-url http://127.0.0.1:8765/whl/cu13/
sha256sum "dist/cache_dit-${V}+cu13-"*.whl .tmp/cache-dit-build-release/*.whl   # 应逐一对齐
```

- 本机 `github.com:443` 被阻断（wheel 资产从 `github.com/.../releases/download/...` 下载），直接 `pip install --index-url https://vipshop.github.io/cache-dit/whl/cu13/` 会在下载资产时卡住；因此线上验收用「本地索引跑通（sha256 与 dist 一致）+ 核对线上页面」两步代替：

```bash
curl -s "https://vipshop.github.io/cache-dit/whl/cu13/cache-dit/" | grep -o 'cache_dit[^"<]*whl'
```

- 安装后自检（在 cdit 或目标 env 里；纯包 `False`，CUDA 包 `True`，版本必须与 wheel 一致）：

```bash
python -c "import cache_dit; from cache_dit.quantization import svdq_is_available; \
print(cache_dit.__version__, svdq_is_available())"
```

- smoke（可选；cache-dit 默认 4/5 号卡）：`CUDA_VISIBLE_DEVICES=4` 跑 `tests/` 下一个轻量用例，确认扩展路径可用。

### 6. 收尾（按需）

- `docs/user_guide/INSTALL.md` 的安装口令同步到 V（`cache-dit==V`、`cache-dit==V+cu13` 只在显式 pin 时出现）；「GLIBC floor 2.38」「SVDQuant 覆盖架构」等注释只在编译集变化时改。
- 验收完可 `git worktree remove .tmp/gh-pages`（或保留复用）。
- 镜像站点（个人站点下的同一套索引，锚点仍指向 vipshop 资产）：

```bash
CACHE_DIT_SITE_URL=https://deftruth.github.io CACHE_DIT_INDEX_PREFIX=cache-dit \
  bash tools/publish_index.sh --index-only --tag "v${V}+cu13" \
  --wheels "dist/cache_dit-${V}+cu13-"*.whl \
  --release-url "https://github.com/vipshop/cache-dit/releases/download/v${V}+cu13" \
  --site-repo <DefTruth.github.io checkout>
```

`--index-only` 必须显式给 `--release-url`（不给会直接 fail）。

## 硬约束（脚本已内置守卫，不要绕过）

| 约束 | 原因 |
|---|---|
| CUDA wheel 必须带 `+cu13` local 段，纯包必须不带 | PyPI 拒收 local 版本；local 段是 pip 区分两种构建的唯一依据 |
| full 档只从已提交源构建 | 产物必须对得上某个 commit |
| mini 档（`.devN`、单 env、单 arch）永不发布 | 只验证流水线；`publish_index.sh` 拒收 `.dev` 与缺 `+cu13` 的 wheel |
| cu13 flavor 必须 `CACHE_DIT_BUILD_SVDQUANT=1` 且 `CACHE_DIT_SVDQUANT_BUILD_MODE=ALL`，nvcc ≥ 13.0 | setup.py `_validate_release_configuration()` 硬校验 |
| 发布 arch 用 setup.py 默认集（`80,86,89,120a,121a`），不收窄 | 脚本定档：full 档 arch 留空即默认集，即发布集；sm90 Hopper 被 SVDQuant INT4 kernel 排除 |
| 索引只收 CPython + manylinux + 编译版 wheel | `publish_index.sh` 逐个断言；纯包/非 manylinux 直接拒收 |
| 安装口径用 `--extra-index-url` 指向自建索引 | 索引只放 `cache-dit` 一个项目、不带依赖，PyPI 必须保持依赖来源 |

## 故障速查

| 症状 | 处置 |
|---|---|
| `profile full builds from committed sources; commit these first` | 提交列出的 shipped inputs，或 `CACHE_DIT_RELEASE_ALLOW_UNCOMMITTED=1` |
| `profile full needs a release version, got '...devN'` | 传 `CACHE_DIT_RELEASE_VERSION=$V`（先给提交打 tag 或显式指定） |
| `the pure-Python release carries a plain version; '...' has a local segment` | full 的 V 不能带 `+`；local 段只属于 CUDA wheel |
| `nvcc is CUDA $toolkit; the cu13 release flavor needs 13.0+` | 把 `CUDA_HOME` 指到 CUDA 13.x |
| `refusing to publish a pre-release wheel` | 误用了 mini 档产物，改用 full 档重编 |
| `tag '...' does not name version '...'` | tag 必须写成 `v${V}+cu13` |
| `already published and cannot be shown to be this artifact` | `CACHE_DIT_FORCE_UPLOAD=1` 替换资产，或换版本号 |
| `this index only publishes compiled wheels` | glob 把纯包也圈进来了；用 `cache_dit-${V}+cu13-*.whl` |
| `does not carry a manylinux platform tag` | 先跑 `tools/repair_wheel_platform_tag.py --retag`（build_releases.sh 通常已做） |
| 线上索引看不到新版本 | GitHub Pages 部署有约 1 分钟延迟，别用 `?cb=` 误判成没发上去 |
| 误发了不该发的更高版本 | 删掉该 Release（资产）**并**删掉索引页里对应锚点（`make_index.py` 是 merge 语义，只删 release 不够），再用正确 wheel 重新生成索引 |

## 参考文件

- `references/release-pipeline.md`：发布工具链（`build_releases.sh` / `build_fast.sh` / `release_workspace.py` / `inspect_release_wheel.py` / `repair_wheel_platform_tag.py` / `make_index.py` / `publish_index.sh`）的职责与关键行为、`CACHE_DIT_RELEASE_*` 与发布注入环境变量全表、守卫清单、mini/full profile 对照、版本推导。
- `references/install-and-index.md`：两种构建的安装语义与 pip 行为、索引与资产布局、验收配方、已发布事实（v1.5.3 / v1.5.3+cu13）与历史决策（为何 CUDA 包不上 PyPI、旧 `cache-dit-cu13` 发行退役）。
