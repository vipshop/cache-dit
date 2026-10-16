# 安装与索引参考

## 1. 两种构建

| 构建 | 内容 | 来源 |
|---|---|---|
| `cache-dit` | 纯 Python 源码 | [PyPI](https://pypi.org/project/cache-dit/) |
| `cache-dit`（CUDA） | 同一份源码编译进 SVDQuant 扩展（`cache_dit._C_svdquant`） | 自建索引 `https://vipshop.github.io/cache-dit/whl/cu13/` |

两者同名同包（`cache_dit/`），靠 wheel 的 local 版本区分：CUDA 包是 `1.5.3+cu13`，纯包是 `1.5.3`。

## 2. 安装命令与语义

```bash
pip install -U cache-dit                                                     # 默认：PyPI 纯包
pip install cache-dit --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/   # CUDA 扩展包
```

- **必须用 `--extra-index-url`（不是 `--index-url`）**：自建索引只放 `cache-dit` 一个项目、不带依赖，PyPI 必须保持依赖来源。`pip` 合并两个索引取最高版本；`uv`（0.11.28 实测）先搜 extra index 再搜默认 index，所以这条命令对两种安装器都解析到 CUDA 构建。反过来写（`--index-url` 指向自建索引、PyPI 作 extra）对 `pip` 可用，但 `uv` 会只考虑 PyPI 的 release，除非再加 `--index-strategy unsafe-best-match`。
- **先装 torch**：wheel 只针对 CUDA 13.x 编译（验证基线 `torch==2.11.0+cu130`）。让 pip 从 PyPI 拉 torch 会拿到非 cu13 构建，装完 `import` 时报 `libcudart.so.13` 找不到。先 `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130`。
- 运行前提：Linux x86_64、glibc ≥ 2.38、Python 3.10–3.14（以索引页列为准）、torch cu130、GPU 属于已编译架构（sm 80/86/89/120a/121a）。
- pip 行为（实测）：local 版本排序高于同 base 版本，两个索引同时可用时 CUDA 包胜出，且已装的纯包不再满足需求；装过 CUDA 包后 `pip install -U cache-dit`（不带自建索引）**不会**降级回纯包；要退回纯包是「卸载再装」：

```bash
pip uninstall -y cache-dit && pip install -U cache-dit
```

- 旧发行 `cache-dit-cu13`（PyPI 最后版本 `1.5.2`）已退役：两个名字发同一个 `cache_dit/` 包，切换前先 `uv pip uninstall cache-dit-cu13`。这一口径写在 `docs/user_guide/INSTALL.md` 的 WARNING 里。

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
mkdir -p .tmp/cache-dit-build-release
python3 -m http.server -d .tmp/gh-pages 8765 &     # 服务本地索引工作树
pip download "cache-dit==${V}+cu13" --no-deps -d .tmp/cache-dit-build-release \
  --index-url http://127.0.0.1:8765/whl/cu13/
sha256sum "dist/cache_dit-${V}+cu13-"*.whl .tmp/cache-dit-build-release/*.whl
```

本地 HTTP 索引的下载会走锚点里的 sha256 校验，字节与 `dist/` 一致即等价于线上可装。
本机 `github.com:443` 被阻断（资产从 `github.com/.../releases/download/...` 下载），所以线上闭环用「本地索引 + 页面核对」：

```bash
curl -s "https://vipshop.github.io/cache-dit/whl/cu13/cache-dit/" | grep -o 'cache_dit[^"<]*whl'
```

安装后自检（纯包 `False`，CUDA 包 `True`；版本必须与 wheel 一致）：

```bash
python -c "import cache_dit; from cache_dit.quantization import svdq_is_available; \
print(cache_dit.__version__, svdq_is_available())"
```

## 5. 已发布事实（2026-10-16 核对，会随时间变化）

- Release `v1.5.3+cu13`：5 个资产（cp310–cp314，`manylinux_2_38_x86_64`，各 ~32 MB），非 Latest；Release `v1.5.3`：Summary 型 notes（release-notes-generator 生成），**无资产**，Latest。
- PyPI `cache-dit` 最新版 `1.5.3`，只有 `cache_dit-1.5.3-py3-none-any.whl`（纯包，0.53 MiB）。
- 两个 tag 并非同一修订：`v1.5.3` → `b7f8276`（#1120，CUDA 线改为同名 + `+cu13`）、`v1.5.3+cu13` → `9109a46`（RPATH `$ORIGIN` 修复，补发形态）。后续发布建议两者同源。
- 本机发布环境：base env `cdit` + `py310`–`py314` 五个 CPython env 均可用；`gh`（DefTruth，GITHUB_TOKEN）、`twine`、`ccache`、`patchelf`、`objdump` 已安装；`CUDA_HOME` 默认 `/usr/local/cuda`（需 CUDA 13.x）。

## 6. 为何 CUDA 包不上 PyPI（决策记录）

- PyPI 拒收带 local 段（`+cu13`）的版本号，服务端直接 400。
- 若去掉后缀以干净版本上传：实测 pip 在「同名同版本、纯包与平台包并存」时优先选平台 wheel，`pip install -U cache-dit` 默认就会装 32 MB 的 CUDA 包，违背「默认最通用」。
- PyPI 不可变（只能 yank）、manylinux 标签表达不了「需要 torch cu13x」——都指向独立索引。
- 历史先例：`1.5.2` 及以前 CUDA 线用独立发行名 `cache-dit-cu13` 发在 PyPI；`b7f8276`（#1120）起改为同名 + local 版本 + 自建索引，旧发行退役。
