# 安装与索引参考

## 1. 两种构建

| 构建 | 内容 | 来源 |
|---|---|---|
| `cache-dit` | 纯 Python 源码 | [PyPI](https://pypi.org/project/cache-dit/) |
| `cache-dit`（CUDA） | 同一份源码编译进单一 CUDA 扩展（`cache_dit._C_ops`，含 SVDQuant/comm/attn/models 四族） | 自建索引 `https://vipshop.github.io/cache-dit/whl/cu13/` |

两者同名同包（`cache_dit/`），靠 wheel 的 local 版本区分：CUDA 包是 `1.5.5+cu13`，纯包是 `1.5.5`。

## 2. 安装命令与语义

```bash
pip install -U cache-dit                                                     # 默认：PyPI 纯包
pip install cache-dit --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/   # CUDA 扩展包
```

- **必须用 `--extra-index-url`（不是 `--index-url`）**：自建索引只放 `cache-dit` 一个项目、不带依赖，PyPI 必须保持依赖来源。`pip` 合并两个索引并取最高版本，所以这条命令会解析到 CUDA 构建。反过来写（`--index-url` 指向自建索引、PyPI 作 extra）虽然也能装上，但依赖的来源会被这条命令一并改写，不要这么写。
- **先装 torch**：wheel 只针对 CUDA 13.x 编译（本机构建/验证基线 `torch==2.15.0+cu132`，四个发布 env 同版本）。让 pip 从 PyPI 拉 torch 会拿到非 cu13 构建，装完 `import` 时报 `libcudart.so.13` 找不到；先按你使用的 CUDA 13 索引装好 torch，再装 cache-dit。
- 运行前提：Linux x86_64、glibc 满足 wheel 的平台标签、Python 3.11–3.14（以索引页列为准）、torch cu13x（本机构建基线 `2.15.0+cu132`）、GPU 属于已编译架构（默认 setup.py 集，当前 sm 89/120a；`sm90` 一律不编，`sm100` 需显式加编）。**glibc 下限不是常量**：1.5.3 是 `manylinux_2_38`，1.5.5 统一成单一 `_C_ops` 后是 `manylinux_2_34`；以索引页/文件名里的标签为准。
- pip 行为（实测）：local 版本排序高于同 base 版本，两个索引同时可用时 CUDA 包胜出，且已装的纯包不再满足需求；装过 CUDA 包后 `pip install -U cache-dit`（不带自建索引）**不会**降级回纯包；要退回纯包是「卸载再装」：

```bash
pip uninstall -y cache-dit && pip install -U cache-dit
```

- 旧发行 `cache-dit-cu13`（PyPI 最后版本 `1.5.2`）已退役：两个名字发同一个 `cache_dit/` 包，切换前先 `pip uninstall cache-dit-cu13`。这一口径写在 `docs/user_guide/INSTALL.md` 的 WARNING 里。

## 3. 资产与索引布局

| 位置 | URL |
|---|---|
| PEP 503 通道根 | `https://vipshop.github.io/cache-dit/whl/cu13/` → 列出 `cache-dit` |
| 项目页（pip 真正读的） | `https://vipshop.github.io/cache-dit/whl/cu13/cache-dit/` |
| 人类 landing | `https://vipshop.github.io/cache-dit/` |
| wheel 资产 | `https://github.com/vipshop/cache-dit/releases/download/v<版本>+cu13/<wheel 文件名>#sha256=<hex>` |
| 镜像（个人站点，锚点仍指向 vipshop 资产） | `https://deftruth.github.io/cache-dit/whl/cu13/cache-dit/` |

- 索引是纯静态 HTML，放在项目 `gh-pages` 分支根（Pages: Deploy from a branch / `gh-pages` / `(root)`），靠 `.nojekyll` 关掉 Jekyll。
- 每个锚点带 `#sha256=`，pip 强制校验：资产被替换或上传不完整会显式失败，而不是装出坏包。
- 一个 CUDA toolkit 一个目录（`whl/cu13/`）；换 toolkit 只改目录与 local 版本后缀，发行名不变。
- 索引生成是 **merge 语义**（以文件名为键）：发新版本保留历史版本锚点，`pip install cache-dit==<旧版本>` 仍可解析；同文件名重发覆盖自己的锚点，不会留下两个 sha256。
- 本机索引 worktree 在 `cache-dit/.tmp/gh-pages`；构建 workspace 在 `cache-dit/.tmp/release_build/{base,cu13}`（`.tmp` 已被 `.gitignore` 忽略）。

## 4. 验收配方

```bash
mkdir -p .tmp/cache-dit-build-release/dl        # 独立子目录，先清空，避免历史 wheel 混进比对
python3 -m http.server -d .tmp/gh-pages 8765 &     # 服务本地索引工作树
pip download "cache-dit==${V}+cu13" --no-deps -d .tmp/cache-dit-build-release/dl \
  --index-url http://127.0.0.1:8765/whl/cu13/
sha256sum "dist/cache_dit-${V}+cu13-"*.whl .tmp/cache-dit-build-release/dl/*.whl
```

本地 HTTP 索引的下载会走锚点里的 sha256 校验，字节与 `dist/` 一致即等价于线上可装。注意 `pip download` 只解析**当前解释器**对应的那一个 cp 轮子；要覆盖四个，per-env 循环或显式 `--python-version/--abi/--platform`。

本机 `github.com:443` 时通时断（资产从 `github.com/.../releases/download/...` 下载，实测先超时重试、随后 ~40 kB/s 拉完），所以线上闭环用「本地索引 sha256 对齐 + 页面锚点对齐」：

```bash
curl -s "https://vipshop.github.io/cache-dit/whl/cu13/cache-dit/" \
  | grep -o 'cache_dit-<V>[^"#]*whl#sha256=[a-f0-9]*'      # 与本地 sha256sum 排序后 diff
```

安装后自检（纯包 `False`，CUDA 包 `True`；版本必须与 wheel 一致）。必须装到**仓库外**的 `--target` 再跑，并断言 `cache_dit.__file__` 落在其中——否则会命中 dev 树的 `src/` 得到假结论：

```bash
pip install --no-deps --target /tmp/verify-${V}/cuda "dist/cache_dit-${V}+cu13-"*.whl
cd /tmp && PYTHONPATH=/tmp/verify-${V}/cuda python -c "import cache_dit; \
from cache_dit.kernels import cuda_ext_is_available, cuda_ext_get_load_error; \
print(cache_dit.__version__, cache_dit.__file__, cuda_ext_is_available(), cuda_ext_get_load_error())"
```

`svdq_is_available()` 在 #1288 之后已不存在。

## 5. 已发布事实（会随时间变化）

**v1.5.5（2026-11-18，最新）**

- Release `v1.5.5+cu13`：4 个资产（cp311–cp314，`manylinux_2_34_x86_64`，各 ~8 MB），非 Latest；Release `v1.5.5`：notes（Overview + Core Highlights + Upgrade Notes），**无资产**，Latest。
- 两个 tag **同源**：`v1.5.5` 与 `v1.5.5+cu13` 都指向 `8f9b28ca`（PR #1351 的 merge commit）。
- PyPI `cache-dit` 最新版 `1.5.5`，只有 `cache_dit-1.5.5-py3-none-any.whl`（纯包，1.38 MiB，sha256 `685ad24e…`）。
- wheel 内 cubin 只有 `sm_89` + `sm_120a`（`cuobjdump -lelf` 可核），比 1.5.3 少了 80/86/121a：统一 `_C_ops` 的谓词是 `>=sm89` 且排除 `sm90`。

**v1.5.3（2026-10-16 核对，历史）**

- Release `v1.5.3+cu13`：5 个资产（cp310–cp314，`manylinux_2_38_x86_64`，各 ~32 MB），非 Latest；Release `v1.5.3`：Summary 型 notes，**无资产**，Latest。
- PyPI 当时最新版 `1.5.3`，纯包 0.53 MiB。
- 两个 tag 并非同一修订：`v1.5.3` → `b7f8276`（#1120，CUDA 线改为同名 + `+cu13`）、`v1.5.3+cu13` → `9109a46`（RPATH `$ORIGIN` 修复，补发形态）。**v1.5.5 起改为同源，不要再照抄补发形态。**

**本机发布环境（两轮共用）**

- base env `cdit` + 发布用的 `py311`–`py314`（`py310` 自 2026-10-30 起不再进发布集，torch 后续版本也不再提供）；`gh`（DefTruth，GITHUB_TOKEN）、`twine`、`ccache`、`patchelf`、`objdump` 已安装；`CUDA_HOME` 默认 `/usr/local/cuda`（CUDA 13.x，1.5.5 为 13.2）。

## 6. 为何 CUDA 包不上 PyPI（决策记录）

- PyPI 拒收带 local 段（`+cu13`）的版本号，服务端直接 400。
- 若去掉后缀以干净版本上传：实测 pip 在「同名同版本、纯包与平台包并存」时优先选平台 wheel，`pip install -U cache-dit` 默认就会装那个 CUDA 包（当时 32 MB，1.5.5 为 ~8 MB），违背「默认最通用」。
- PyPI 不可变（只能 yank）、manylinux 标签表达不了「需要 torch cu13x」——都指向独立索引。
- 历史先例：`1.5.2` 及以前 CUDA 线用独立发行名 `cache-dit-cu13` 发在 PyPI；`b7f8276`（#1120）起改为同名 + local 版本 + 自建索引，旧发行退役。
