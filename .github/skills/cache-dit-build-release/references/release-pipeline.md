# 发布链路内部实现参考

代码路径均相对 cache-dit 仓库根目录。行为以当前工作树（`dev` 分支）为准。

## 1. 工具链总览

| 工具 | 角色 | 关键行为 |
|---|---|---|
| `tools/build_releases.sh` | 发布入口 | 解析 profile 与版本 → 准备 `base`/`cu13` 两个 workspace → base env 出纯 wheel → 逐 env 出 cu13 wheel → `repair_wheel_platform_tag.py` 归一 → `inspect_release_wheel.py` 断言 → 打印 twine/publish 命令 |
| `tools/build_fast.sh` | 编译开关解析 + 加速 | ccache 影子 `CUDA_HOME`（只替换 `bin/nvcc` 为 `tools/nvcc` 包装器）；`MAX_JOBS` 默认 32；`CACHE_DIT_CLEAN=1` 清 `build/`；末尾 `python -m pip wheel . --no-build-isolation --no-deps -w dist` |
| `tools/release_workspace.py` | 出树 workspace | 拷贝 `setup.py`+`src`（排除 `*.so`/`_version.py`/`__pycache__`/`*.egg-info`），软链 `csrc`、`.git` 与两个工具；setuptools-scm 在 workspace 内写 `_version.py`，不碰开发树 |
| `tools/inspect_release_wheel.py` | 产物断言 | `--so none`（纯包不得含 `.so`）/ `--so svdquant`（CUDA 包必须含 SVDQuant、comm、attn 三个扩展）；`_version.py` 与 METADATA 版本必须一致 |
| `tools/repair_wheel_platform_tag.py` | 产物归一 | 扫 ELF 的 `GLIBC_x.y` 定 manylinux 下限（当前 2.38）→ 改文件名与 WHEEL 标签；`--set-rpath '$ORIGIN'`；重算 RECORD 重打包 |
| `tools/make_index.py` | 索引生成 | PEP 503 三层页面（通道根 → 项目页 → landing）+ `.nojekyll`；merge 语义（按文件名覆盖，历史保留）；锚点 = Release 资产 URL + `#sha256=` |
| `tools/publish_index.sh` | 发布编排 | `gh release create`（`--latest=false`、`--target HEAD`、notes 写安装命令）→ 资产上传（幂等）→ `make_index.py` → commit/push 站点仓库 |

`build_releases.sh` 与 `publish_index.sh` 的用法在各自 shebang 注释里（`--help` / 文件头即手册），改动前先读。

## 2. `CACHE_DIT_RELEASE_*` 环境变量（build_releases.sh）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CACHE_DIT_RELEASE_PROFILE` | `mini` | `mini` / `full` |
| `CACHE_DIT_RELEASE_VERSION` | 空（scm 推导） | full 必给；mini 禁止给（版本必须由 scm 推导并强制 `.dev`） |
| `CACHE_DIT_RELEASE_ENVS` | mini `py312` / full `py311 py312 py313 py314` | 空格分隔的 conda env |
| `CACHE_DIT_RELEASE_DIST` | `dist/` | 产物落点 |
| `CACHE_DIT_RELEASE_WORK_ROOT` | `.tmp/release_build` | workspace 根（`base/` 与 `<channel>/` 两个子目录） |
| `CACHE_DIT_RELEASE_CHANNEL` | `cu13` | local 版本后缀 + 索引目录 |
| `CACHE_DIT_RELEASE_ARCH` | 空 = setup.py 默认全架构集（发布必须留空） | mini 默认单 arch（`CACHE_DIT_RELEASE_MINI_ARCH`，默认 `120a`） |
| `CACHE_DIT_RELEASE_BASE_ENV` | `cdit` | 纯 wheel 与所有归一/断言步骤所在 env |
| `CACHE_DIT_RELEASE_JOBS` | 空 | → `MAX_JOBS` |
| `CACHE_DIT_RELEASE_ALLOW_UNCOMMITTED` | `0` | `1` 允许脏树构建（告警，产物不对应任何 commit） |
| `CACHE_DIT_RELEASE_DRY_RUN` | `0` | `1` 只解析并打印，不构建 |
| `CACHE_DIT_RELEASE_NAME` | `cache_dit` | 发行名（wheel stem） |

cu13 wheel 构建时脚本固定注入：`CACHE_DIT_RELEASE_FLAVOR=cu13`、`CACHE_DIT_BUILD_SVDQUANT=1`、
`CACHE_DIT_BUILD_WHEELS=1`、`CACHE_DIT_SVDQUANT_BUILD_MODE=ALL`、`CACHE_DIT_WHEEL_PLATFORM_TAG=linux_x86_64`、
`CACHE_DIT_CUDA_ARCH_LIST`、`TORCH_CUDA_ARCH_LIST=`（清空，防止 torch 的 arch 列表渗入）、
`CACHE_DIT_REQUIRE_CCACHE=1`、`CACHE_DIT_CLEAN=1`。纯 wheel 只注入 `SETUPTOOLS_SCM_PRETEND_VERSION=$V`。
脚本还会 unset `CACHE_DIT_VERSION_WRITE_TO` / `CACHE_DIT_BUILD_LOCAL_VERSION`，防止版本文件写回 checkout
或纯包被加 local 段。

## 3. `publish_index.sh` 参数与环境

| 项 | 默认 | 说明 |
|---|---|---|
| `--tag` | 必给 | `v<V>+cu13`，与 wheel local 版本同构 |
| `--wheels` | 必给 | 编译版 wheel 路径或 glob（shell 会先展开成多个参数） |
| `--site-repo` | 必给 | gh-pages worktree 根（`--index-only` 时可为任意目录） |
| `--release-url` | `--index-only` 时必给 | 资产 URL 前缀 |
| `--index-only` / `--no-push` | 关 | 只生成索引 / 只 commit 不 push |
| `CACHE_DIT_RELEASE_REPO` | `vipshop/cache-dit` | 资产所在仓库 |
| `CACHE_DIT_SITE_URL` | `https://vipshop.github.io/cache-dit` | 站点根 |
| `CACHE_DIT_INDEX_PREFIX` | 空 | 项目 Pages 分支为空（仓库名即 URL 路径）；镜像站点设 `cache-dit` |
| `CACHE_DIT_INDEX_PROJECT` | `cache-dit` | 索引的发行名 |
| `CACHE_DIT_INDEX_CHANNEL` | `cu13` | 通道子目录 |
| `CACHE_DIT_FORCE_UPLOAD` | `0` | `1` = `--clobber` 替换同名资产 |

## 4. 守卫清单（照抄自脚本，改动前先看这里）

`build_releases.sh`：

- 每个 env：Python 版本与 `py<M><m>` 名匹配；能 `import packaging, setuptools_scm, torch, wheel`。
- 任意 profile：`$CUDA_HOME/bin/nvcc` 存在且 major ≥ 13（cu13 flavor 的硬性要求）。
- full：shipped inputs（`src csrc tools setup.py pyproject.toml setup.cfg MANIFEST.in README.md LICENSE`）无未提交改动；版本不含 `.dev`、不含 `+`；版本形如 `x.y.z`。
- mini：拒绝 `CACHE_DIT_RELEASE_VERSION`；scm 给出的裸版本强制补 `.dev0`。
- 每个 env 断言 dist 里「恰好一个」该 CPython tag 的 cu13 wheel；cu13 wheel 总数必须等于 env 数；纯 wheel 恰好一个。
- 构建前删掉 dist 里同版本旧 wheel，避免旧产物顶替断言。
- 校验每个 cu13 wheel 的文件名平台标签 = 检测到的 manylinux 下限、RPATH = `$ORIGIN`。

`setup.py`：

- `_validate_release_configuration()`：cu13 flavor 必须 `CACHE_DIT_BUILD_SVDQUANT=1` + `CACHE_DIT_SVDQUANT_BUILD_MODE=ALL` + nvcc ≥ 13.0。
- SM 目标：`DEFAULT_CUDA_ARCH_TARGETS = ("89", "90a", "100", "120a")` 是默认全架构集，`_default_cuda_arch_targets()` 按 nvcc 版本门限裁剪（sm90a ≥ 12.0，sm100 / sm120a ≥ 12.8；CUDA 13 下四个全在）。显式 arch 列表（`CACHE_DIT_CUDA_ARCH_LIST` 优先于 `TORCH_CUDA_ARCH_LIST`）会覆盖默认集。
- 三个扩展各自过滤：SVDQuant 走 `_is_supported_svdquant_sm_target`（≥ sm80 且剔除 sm90），comm 与 attn 走 `_get_ops_sm_targets(scope)` → `_is_supported_comm_sm_target`（≥ sm89、保留 sm90）——同一次构建里各扩展的 gencode 集合可以不同，这是预期行为。cu13 flavor 下任一扩展解析出空 SM 集会直接 raise，不会静默出一个缺 `.so` 的 wheel。

`publish_index.sh`：

- 拒收：非 `cache_dit-` 前缀、`-any.whl`（纯包）、非 `-cp*-cp*-`、非 `manylinux_`、`.dev` 版本、缺 `+`、local 段非 `cu13`、一次发布混版本、tag 与版本不符。
- 同名资产：size 相同且（API digest 或索引页 sha256）与本地一致才跳过；否则要求 `CACHE_DIT_FORCE_UPLOAD=1`。
- release 不存在时创建：`--target` 取 cache-dit 仓库当前 HEAD（必须已 push）；`--latest=false` 不让通道 release 抢占 Latest。
- `--site-repo` 必须是 git work tree 根（`--index-only` 例外）。
- commit 用 `PRE_COMMIT_ALLOW_NO_CONFIG=1`（worktree 共享源仓库的 pre-commit 钩子，索引树没有自己的配置）。
- `--index-only` 必须显式给 `--release-url`（不给直接 fail）。

## 5. mini / full profile 对照

| profile | envs | arch | 版本 | 产物 | 用途 |
|---|---|---|---|---|---|
| `mini` | `py312` | 单 arch（默认 `120a`） | scm 推导 `1.5.x.devN` | 纯 wheel + 1 个 cu13 wheel | 流水线验证；禁止发布 |
| `full` | `py311`–`py314` | setup.py 默认全架构集（CUDA 13 下 `89,90a,100,120a`；arch 留空） | 必给 V | 1 纯 wheel + 4 cu13 wheel（各 ~32 MB） | 发布 |

## 6. 版本推导

- 不传 `CACHE_DIT_RELEASE_VERSION` 时用 setuptools-scm：`version_scheme='python-simplified-semver'`、`local_scheme='no-local-version'`；mini 对裸版本补 `.dev0`（仍排在该正式版之下，不会被误认）。
- 构建时以 `SETUPTOOLS_SCM_PRETEND_VERSION` 钉版本（cu13 为 `$V+cu13`），由 workspace 内的 `_version.py` 承载。
- tag：CUDA 通道 `v<V>+cu13` 由 `publish_index.sh` 的 `gh release create` 自动创建；普通 release `v<V>` 在第 4 步单独创建（PyPI 上传之后）。
