---
name: cache-dit-build-release
description: "cache-dit 发布全流程（给定一个版本号即可执行完）：full profile 一次构建 PyPI 纯 Python wheel 与带单一 `_C_ops` CUDA 扩展的 +cu13 wheel（cp311–cp314，默认编译 setup.py 默认 arch 集——当前 sm 89/120a，需覆盖 Blackwell datacenter 时显式加 100，禁止用 CACHE_DIT_RELEASE_ARCH 收窄，manylinux 平台标签与 $ORIGIN RPATH 归一）、gh-pages 上的 PEP 503 静态索引发布、GitHub Release 资产上传、PyPI 上传，以及发布后验收与安装语义。Use when cutting a cache-dit release, building or publishing the CUDA-extension wheels under whl/cu13, regenerating the wheel index on gh-pages, uploading the pure-Python wheel to PyPI, or verifying an install from the index."
user-invocable: true
---

# cache-dit 发布（build & release）

一次发布 = 同一个版本号下产出两件东西：**PyPI 的纯 Python 包**与**自建索引上的 SVDQuant CUDA 扩展包**。
两者同名同包（`cache_dit/`），靠 wheel 的 local 版本区分：CUDA 包是 `1.5.5+cu13`，纯包是 `1.5.5`。
除版本号 `V`（如 `1.5.5`）外无需其它输入；通道 `CHANNEL` 默认 `cu13`，除新增 CUDA toolkit 外不要改。
下文所有路径相对 **cache-dit 仓库根目录**（本机 `/workspace/dev/vipshop/cache-dit`）。

> **默认编译全架构（硬性要求，最高优先级）**
> 发布用的 CUDA wheel 必须用 `setup.py` 的默认 arch 集构建，也就是**不设** `CACHE_DIT_RELEASE_ARCH`（留空）。
> （mini 档的单 arch 仅用于流水线验证、永不发布，不在此列。）
> `setup.py` 的 `DEFAULT_CUDA_ARCH_TARGETS` 现在是 `("89", "120a")`（Ada + Blackwell
> consumer，每个额外 arch 都成倍拉长单扩展编译时间），再按 nvcc 版本门限裁剪与谓词过滤，即发布集。
> 不要手写 arch 列表：手写会随 setup.py 漂移，而**少编一个 arch**会让该卡用户装上直接
> `no kernel image is available for execution on the device`；要覆盖 Blackwell datacenter 就在
> `CACHE_DIT_RELEASE_ARCH` 之外显式要求加 `100`（当前默认不含）。
> 统一谓词是 `>=sm89` 且剔除 `sm90`（Hopper 一律不编），这些差异由 `setup.py` 负责，发布层不干预。

## 产物清单（以 `${V}` 为例）

| 产物 | 名称 | 去向 |
|---|---|---|
| CUDA wheel ×4 | `cache_dit-${V}+cu13-cp3{11..14}-cp3{11..14}-manylinux_<GLIBC>_x86_64.whl`（各 ~8 MB） | Release `v${V}+cu13` 资产 + gh-pages 索引 `whl/cu13/` |
| 纯 Python wheel ×1 | `cache_dit-${V}-py3-none-any.whl`（~1.4 MB） | PyPI（Release `v${V}` 挂 notes；资产可附可不附，见第 4 步） |
| 索引页 | `whl/cu13/cache-dit/index.html`（PEP 503，每个锚点带 sha256） | 项目 `gh-pages` 分支根 → `vipshop.github.io/cache-dit/` |

`<GLIBC>` 由 `tools/repair_wheel_platform_tag.py` 按扩展**实际**需要的 GLIBC 符号推导，不是常量：统一成单一 `_C_ops` 后 1.5.5 是 `manylinux_2_34_x86_64`（1.5.3 是 `2_38`）。体积同理，1.5.5 的 `.so` 约 20 MB（1.5.3 的 `_C_svdquant` 是 137 MB）。要写进 release notes 就按当轮实际产物写，别沿用旧数字。

安装口径（发布后用户侧）：

```bash
pip install -U cache-dit                                                     # 默认：PyPI 纯 Python 包
pip install cache-dit --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/   # CUDA 扩展包，如 1.5.5+cu13
```

注意与 ffpa-attn 的差异：cache-dit 的自建索引只放 `cache-dit` 一个项目、不带依赖，因此必须用
`--extra-index-url`（PyPI 保持依赖来源），且要先装 CUDA 13 构建的 torch，详见
`references/install-and-index.md`。

## 执行顺序（勿跳步）

> **两条顺序纪律（v1.5.5 复盘）**
> 1. **构建先于 tag 与公开 Release**。`build_releases.sh` 用 `SETUPTOOLS_SCM_PRETEND_VERSION` 注入版本，
>    **不依赖 tag 存在**；先打 tag 只是把不可逆动作（公开 tag + Latest Release）放在 4 个 env 的编译之前，
>    一旦编译或产物断言失败，只能删 tag 或被动换版本号。推荐顺序：
>    `PR merge → build → 产物断言 → git tag -a v<V> -m ... → push tag → gh release create v<V> → publish_index → twine`。
> 2. **`publish_index.sh` 的 `--target` 取的是"运行时的仓库 HEAD"**，不是 `v<V>` tag。跑它之前必须断言
>    `git rev-parse HEAD == git rev-parse v<V>^{commit}`（且该 commit 已 push），否则 `v<V>+cu13` 会指到别的提交
>    ——`v1.5.3` 的分叉就是这么来的。

### 0. 预检

```bash
git status --porcelain -- src csrc tools setup.py pyproject.toml setup.cfg MANIFEST.in README.md LICENSE
"${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" --version | tail -2     # 需 CUDA 13.x（cu13 flavor 硬性要求）
for e in py311 py312 py313 py314; do \
  conda run -n "$e" python -c \
  "import sys, packaging, setuptools_scm, torch, wheel; print(sys.version.split()[0], torch.__version__)"; done
command -v ccache patchelf objdump gh twine
gh auth status
gh release view "v${V}" --repo vipshop/cache-dit                 # 期望 not found
gh release view "v${V}+cu13" --repo vipshop/cache-dit            # 期望 not found
```

- 第一条输出必须为空：full 档只从已提交源构建。确有需要可 `CACHE_DIT_RELEASE_ALLOW_UNCOMMITTED=1` 绕过（会告警，产物不对应任何 commit）。
- PyPI 版本不可覆盖：确认 `cache-dit` 上没有 V（`curl -s https://pypi.org/pypi/cache-dit/json`）。
- py311–py314 提供发布用的 CPython 集合（Python 3.10 已不再支持，torch 后续版本也不再提供），缺哪个就少一个 wheel；发布集合以实际 env 为准。本机 4 个 env 均已存在（`conda env list`）。
- gh-pages worktree：仓库常备 `.tmp/gh-pages`（`git worktree list` 可见），直接复用；不存在则 `git worktree add .tmp/gh-pages gh-pages`。

### 1. dry-run（可选，首次发布或改过发布脚本时）

```bash
CACHE_DIT_RELEASE_PROFILE=full CACHE_DIT_RELEASE_VERSION=$V CACHE_DIT_RELEASE_DRY_RUN=1 \
  bash tools/build_releases.sh
```

### 2. 构建全部产物（full 档：纯 wheel + CUDA wheel 一次出齐）

```bash
CACHE_DIT_RELEASE_PROFILE=full CACHE_DIT_RELEASE_VERSION=$V CACHE_DIT_RELEASE_CHANNEL=cu13 \
  CACHE_DIT_RELEASE_ENVS="py311 py312 py313 py314" \
  bash tools/build_releases.sh
```

- 与 ffpa-attn 不同：cache-dit 没有单独的 pure profile。full 档在同一个 base env（默认 `cdit`）里先出纯 wheel，再逐 env 出 cu13 wheel，最后统一校验（纯包无 `.so`、CUDA 包含唯一的 `_C_ops`（SVDQuant/comm/attn/models 四族共用）、平台标签与 `$ORIGIN` RPATH）。
- **arch 一律留空 = 走默认集**：不传 `CACHE_DIT_RELEASE_ARCH`，由 `setup.py` 的默认集（`DEFAULT_CUDA_ARCH_TARGETS` / `_default_cuda_arch_targets()`，当前为 `89,120a`，按 nvcc 版本门限裁剪后再过谓词）决定，这就是发布集。不要手写一份 arch 列表（会随 setup.py 漂移）；要加 Blackwell datacenter 必须显式把 `100` 加进默认集或 arch 列表，否则 `100` 机器上 `no kernel image`。
- 重发同版本：无 incremental 开关，但 ccache 命中后重编很快；workspace 在 `.tmp/release_build/{base,cu13}`，删掉即可全新重来。
- 只做回归验证、不碰已发布件：`CACHE_DIT_RELEASE_DIST=.tmp/<task>/dist`。
- 编译耗时较长（ccache 无命中时更久）：发出命令后等它结束，不要轮询。1.5.5 实测 4 个 env 共约 12 分钟（ccache 有命中）。
- **产物断言通过才打 tag**：`git tag -a "v${V}" -m "v${V}" && git push origin "v${V}"`。第 3 步的 `HEAD == v<V>^{commit}` 断言需要这个 tag 已存在，所以它必须在这里完成，不能留到第 4 步。

### 3. 发布 CUDA 通道（Release 资产 + gh-pages 索引）

```bash
git -C .tmp/gh-pages pull --rebase          # 复用已有 worktree 时先同步
bash tools/publish_index.sh --tag "v${V}+cu13" \
  --wheels "dist/cache_dit-${V}+cu13-"*.whl --site-repo .tmp/gh-pages
```

一次调用做完四件事：`gh release create v${V}+cu13`（`--latest=false`、`--target` 当前 HEAD、notes 写 `--extra-index-url` 安装命令）→ 上传全部 wheel 资产（幂等：size + digest/索引 sha256 一致才跳过）→ 按 merge 语义重生成索引（历史版本保留）→ commit 并 push `gh-pages`。

- **跑之前断言 HEAD 就是发布提交**：`publish_index.sh` 用 `git rev-parse HEAD` 作为 Release 的 `--target`，不读 `v<V>` tag；``[[ "$(git rev-parse HEAD)" == "$(git rev-parse v<V>^{commit})" ]]`` 不成立就停下（见「两条顺序纪律」）。
- tag 规范 `v<版本>+<通道>`，与 wheel 的 local 版本同构；不一致会直接报错。
- 同名资产已存在且内容不同：脚本拒绝，需 `CACHE_DIT_FORCE_UPLOAD=1` 显式替换（或换版本号）。
- push 被拒时按打印的提示处理：`git -C .tmp/gh-pages pull --rebase && git -C .tmp/gh-pages push`。

### 4. 上传纯包（PyPI + 普通 Release）

```bash
python -m twine upload "dist/cache_dit-${V}-py3-none-any.whl"
gh release create "v${V}" --repo vipshop/cache-dit --verify-tag \
  --notes-file <notes.md>
```

- release notes 用 `release-notes-generator` skill 从 git 历史生成，与历史 release 的 Summary 形态一致。
- `--verify-tag` 让 gh 只认第 2 步末已经推上去的 `v${V}`，不靠它的默认行为另建 tag。
- twine 403 表示 token 没有该项目权限。
- `gh release create` 不带资产是本仓惯例（`v1.5.3`、`v1.5.5` 都只挂 notes，纯包以 PyPI 为准）；要附就确保与 PyPI 上是同一个文件。
- 纯包与 CUDA 通道必须来自**同一个发布提交**：`v1.5.5` 与 `v1.5.5+cu13` 都指向 `8f9b28ca`。历史上的 `v1.5.3` 是补发形态（`v1.5.3` → `b7f8276`、`v1.5.3+cu13` → `9109a46`），**不要再照抄**。

### 5. 发布后验收（必做）

```bash
mkdir -p .tmp/cache-dit-build-release/dl
python3 -m http.server -d .tmp/gh-pages 8765 &          # 本地服务索引工作树
pip download "cache-dit==${V}+cu13" --no-deps \
  -d .tmp/cache-dit-build-release/dl --index-url http://127.0.0.1:8765/whl/cu13/
sha256sum "dist/cache_dit-${V}+cu13-"*.whl      # 与下载到的那个逐个比
```

- **一个 env 只验证得了一个 cp 轮子**：`pip download` 按当前解释器解析（cdit = cp312）。要覆盖 4 个就 per-env 循环，或加 `--only-binary :all: --implementation cp --python-version 3.1x --platform manylinux_2_34_x86_64 --abi cp31x`。下载目录要先清空，否则历史残留会混进 `*.whl` 比对。
- 本机 `github.com:443` 时通时断（资产从 `github.com/.../releases/download/...` 下载）：实测先超时重试，随后以 ~40 kB/s 拉完，8 MB 约 5 分钟。所以线上闭环仍用「本地索引 sha256 对齐 + 线上页面锚点对齐」，不要靠 `pip install --index-url https://...` 计时：

```bash
curl -s "https://vipshop.github.io/cache-dit/whl/cu13/cache-dit/" \
  | grep -o 'cache_dit-<V>[^"#]*whl#sha256=[a-f0-9]*'      # 与本地 sha256sum 排序后 diff
gh api repos/vipshop/cache-dit/releases \
  --jq '.[] | select(.tag_name|startswith("v<V>")) | "\(.tag_name) target=\(.target_commitish) assets=\(.assets|length)"'
```

- PyPI 侧不必整包下载，用 JSON 摘要比对：`curl -s https://pypi.org/pypi/cache-dit/<V>/json` 取 `urls[].digests.sha256`，与 `sha256sum dist/cache_dit-<V>-py3-none-any.whl` 一致即可。
- 安装自检必须在**仓库外**目录做（`--target` + `PYTHONPATH`，cwd 不在 repo 根），否则会命中 dev 树的 `src/` 得到假结论；先断言 `cache_dit.__file__` 落在 target 内。纯包 `False`、CUDA 包 `True`，版本与 wheel 一致：

```bash
pip install --no-deps --target /tmp/verify-<V>/cuda "dist/cache_dit-<V>+cu13-"*.whl
cd /tmp && PYTHONPATH=/tmp/verify-<V>/cuda python -c "import cache_dit; \
from cache_dit.kernels import cuda_ext_is_available, cuda_ext_get_load_error; \
print(cache_dit.__version__, cache_dit.__file__, cuda_ext_is_available(), cuda_ext_get_load_error())"
```

- `svdq_is_available()` / `svdq_extension_is_available()` 已不存在（#1288 收敛为单一 `cuda_ext_is_available()` / `cuda_ext_get_load_error()`），照旧写会 `ImportError`。注意 `cuda_ext_is_available()` 只说明"编进来了"：架构不在 cubin 集合里的卡要到第一次 kernel launch 才失败。
- smoke（可选；cache-dit 默认 4/5 号卡）：`CUDA_VISIBLE_DEVICES=4` 跑一次 `fp8_quantize` + `fp8_gemm`（或 `tests/` 下轻量用例），确认扩展真的能跑，而不只是能 import。

### 6. 收尾（按需）

- `docs/user_guide/INSTALL.md` 的安装口令同步到 V（`cache-dit==V`、`cache-dit==V+cu13` 只在显式 pin 时出现）。GLIBC floor 不要写死：它由 `repair_wheel_platform_tag.py` 从符号推导（1.5.3 是 2.38、1.5.5 统一扩展后是 2.34），以索引页为准。
- **架构覆盖变化必须写进 release notes 的 Upgrade Notes**：统一 `_C_ops` 后谓词是 `>=sm89` 且排除 `sm90`，所以 1.5.5 的 cubin 只有 `sm_89`/`sm_120a`；Ampere（sm_80/86）在 1.5.3 还有 cubin、1.5.5 起没有，Hopper（sm_90）则一直不在发布集里——两者都是装得上但跑不了。`cuobjdump -lelf` 是核对手段：解出 wheel 里的 `.so` 后应看到 `sm_89.cubin` 与 `sm_120a.cubin`（1.5.3 的 `_C_svdquant` 里还有 80/86/121a）。
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
| cu13 flavor 必须 `CACHE_DIT_BUILD_CUDA=1` 且 `CACHE_DIT_CUDA_BUILD_MODE=ALL`，nvcc ≥ 13.0 | setup.py `_validate_release_configuration()` 校验开关与 mode，nvcc 门限由 `_require_min_cuda_for_cu13_release()` 校验 |
| 发布 arch 必须留空，走 setup.py 默认集（当前 `89,120a`） | `setup.py` 的 `DEFAULT_CUDA_ARCH_TARGETS` 即默认集，谓词 `>=sm89` 且剔除 `sm90` 会再过滤一遍；需覆盖 Blackwell datacenter 时要显式加 `100` |
| cu13 下缺扩展即 fail | 只有一个扩展：空 SM 集（≥sm89 且排除 sm90 之外没有目标）在 base/cu13 都直接 raise，不会产出缺 `.so` 的 wheel；产物侧由 `inspect_release_wheel.py --so svdquant` 断言 `_C_ops` 存在 |
| 索引只收 CPython + manylinux + 编译版 wheel | `publish_index.sh` 逐个断言；纯包/非 manylinux 直接拒收 |
| `v<V>` 与 `v<V>+cu13` 必须同源（同一个 commit） | 两者都是用户可见的发布点；`publish_index.sh` 用运行时 HEAD 作 `--target`，跑之前断言 `HEAD == v<V>^{commit}` |
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
- `references/install-and-index.md`：两种构建的安装语义与 pip 行为、索引与资产布局、验收配方、已发布事实（v1.5.5 与历史 v1.5.3）与历史决策（为何 CUDA 包不上 PyPI、旧 `cache-dit-cu13` 发行退役）。
