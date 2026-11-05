# 从上游仓库移植算子（operator migration）

> 本文原为独立 skill `cache-dit-operator-migration`，2026-11-05 合并进 `cache-dit-cuda-kernel`。
> 读它之前先读 `kernel-integration-chain.md`（本仓的层级契约、模板与构建触点）；本文只补"从别的仓库搬算子"这一层的判断与流程。

## 何时读本文

需要用本文的场景：

- 从别的仓库往 cache-dit 搬一个 CUDA 算子
- 把 nunchaku 的算子或 kernel 家族移植进 cache-dit
- 新增一个 CUDA 算子域（新的 native 目录 + 对应 Python 模块）
- 判断一次移植到底必须搬哪些 native 文件
- 为刚移植的算子设计 cache-dit 的公开封装
- 审查一份移植计划，判断它是"想清楚了"还是"机械照搬"

不要用本文的场景：

- 与算子/内核无关的普通模型接入
- 与 kernel、扩展都无关的纯 Python 功能开发
- Triton 或 CuTe DSL 算子开发——本仓不维护这两类
- 无脑"把上游 `csrc` 整段拷进来"的执行

## 目标

把一个算子或 kernel 家族搬进 cache-dit 时，要同时满足：

- 语义正确
- 与 cache-dit 仓库约定一致
- 可选 native 扩展缺失时仍然能安全 import
- 有分层验证，而不是一个 smoke test 就宣布完成

## 1. 后端政策：只维护 CUDA

这是长期的硬约束，不是某个任务的偏好。

cache-dit **只维护 CUDA 算子**。长期留在仓库里的算子都是 `csrc/kernels/` 下的 CUDA kernel，通过 `cache_dit_cuda_ops` 命名空间里的 `torch.library` schema 暴露。

因此：

1. 不要往 cache-dit 里加 Triton 或 CuTe DSL 算子。不做快速路径，不做可移植兜底，不做"先给 A100 用"，也不做临时方案。
2. 不要重建 `src/cache_dit/kernels/triton/`、`src/cache_dit/kernels/cutedsl/` 或任何按后端拆分的子包。这些树已被删除，且由回归守卫盯着，见 `tests/kernels/test_ops_dispatch.py`。
3. 不要重新引入后端选择层。`kernels/backend.py`、`kernels/ops.py`、`KernelBackend` 枚举以及 `_resolve_backend` / `_select_*_backend` 都已删除，`test_ops_dispatch.py` 断言它们不会回来。单一 CUDA 路径 + 清晰的错误信息，取代原来的分派表。
4. 如果上游只有 Triton 实现，读它拿算法，然后用 CUDA 重写 kernel。把 Triton 当伪代码，永远不要当目标语言。
5. 如果有人要 Triton 或 CuTe DSL 支持，明确说明维护政策排除了它们，并给出 CUDA 路线。

理由是维护成本：三套并行后端 = 三套注册层 + 三套 fake 注册路径 + 三套构建故事，收益抵不过成本，而发布 wheel 里只有 CUDA 路径。

仓外独立的 kernel 工作请用 `triton-kernel` 或 `cute-dsl-kernel`，不要把那些工作流带进本仓。

## 2. 核心规则

**不要机械复刻上游结构**。把上游仓库当作语义参考，而不是必须遵守的目录布局。

动代码之前先回答这五个问题：

1. 哪些行为是必须保住的？
2. 保住这些行为所需的最小 native + Python 闭包是什么？
3. 哪些名字要保持与上游源码兼容，哪些应该改成 cache-dit 的命名约定？
4. 哪些必须公开，哪些应当留在私有实现里？
5. 哪些测试能证明移植成功，而不只是"能编译"？

这五个问题答不出来，就不要开始拷文件。

### 引用路径规则

只写可移植的引用：

- 引用 cache-dit 自身文件时写仓内相对路径，例如 `src/cache_dit/kernels/_comm.py`、`tests/kernels/test_cuda_fp8_comm_ops.py`。
- 引用同级或外部仓库时写仓库相对、可被 GitHub 搜索到的路径，例如 `nunchaku/nunchaku/models/linear.py`、`deepcompressor/deepcompressor/backend/nunchaku/utils.py`。
- 不要把 `/abs/path/to/workspace/...` 这类本机绝对路径写进 skill、配套文档，或你为 cache-dit 新增的任何错误信息。

## 3. Phase 0：动手前先收集

### 必需输入

1. 源算子与源仓库。例：`nunchaku/nunchaku/ops/gemm.py` 以及它依赖的 native 文件。
2. cache-dit 侧的目标公开面。例：只做底层 op 封装、只做量化模块，或两者都要。
3. 必需的 dtype、架构与范围边界。例："当前必须交付 INT4 CUDA；FP4 实现可以保留但不作为本轮验收门禁。"
4. 构建与打包要求。例：该算子归哪个扩展、由哪个开关启用、是否需要 submodule。
5. 验收目标。例：import 安全、底层 parity、模块 parity、端到端推理，或非法形状拒绝。

### 收集清单

- 找出源算子的入口点。
- 找出实现这些入口的 native 文件。
- 找出实现真正必需的辅助文件。
- 判断这批行为应该落在哪个既有域模块里。
- 确定必须先跑通的最小功能切片。
- 明确列出本轮**不做**验证的部分。

## 4. Phase 1：调研两侧现状

改动前两边都要看。

### 调研源实现

重点看：

- 从公开 API 到 kernel launch 的真实调用链
- 需要的辅助头文件、互操作层、dispatch 工具与 packer
- 运行期假设：shape、rank、对齐、架构、dtype 限制
- 依赖假设：vendored 头、submodule、环境变量
- 已有测试里值得保留的行为契约
- 上游是否本来就是 CUDA；不是的话记下"kernel 必须重写"

### 调研 cache-dit 侧接入点

锚点文件：

- `src/cache_dit/kernels/__init__.py` —— 公开面
- `src/cache_dit/kernels/_<domain>.py` —— 承载该算子的域模块
- `csrc/kernels/<domain>/` —— native 侧
- `setup.py` —— `_required_<ext>_sources()`、`_missing_<ext>_sources()`、`_get_<ext>_extension()`、`_get_extensions()`
- `tests/kernels/` —— 算子测试
- `tools/inspect_release_wheel.py` —— 发布 wheel 的内容门禁

调研时问自己：

1. 哪个既有域模块是合适的宿主，还是确实需要一个新域？
2. 该算子应该进 `cache_dit._C_comm`、`cache_dit._C_attn`，还是新建/独立扩展？
3. 扩展不可用时，加载失败的形态应该是什么？
4. `cache_dit_cuda_ops` 里这个算子家族是否已有命名约定？
5. `test_ops_dispatch.py` 里是否已经列了本次要扩展的公开名？

## 5. Phase 2：先定移植形态

在动文件之前把设计定下来。

### 1. 先冻结公开面

尽早确定 cache-dit 面向用户的 API：

- 哪些算子名要公开？
- 公开面只做底层、只做模块层，还是两者都要？
- 内部后端开关要对用户隐藏吗？
- 封装函数要显式定义而不是 `partial(...)`，好让编辑器和类型工具看到真实签名吗？

默认规则：调参细节和 launch 细节保持私有，除非有明确的用户需求。

### 2. 只搬最小可用闭包

只需要一个切片时，不要整段搬子系统。

通常要搬：

- 真正在调用链上的 kernel 实现文件
- 它们必需的最少辅助头或 Python 工具
- binding TU 入口、loader、注册与封装

暂时不要搬：

- 上游同目录里与本次无关的 kernel
- 本轮用不到的优化分支
- 不在执行路径上的额外工具、benchmark harness 或框架抽象

### 3. 先保语义，再清理

第一轮移植：

- 先保行为
- 先保 shape 与 dtype 规则
- 先保数据流

不要在同一轮里顺手做命名润色、API 重构或算法改动，除非它们是仓库一致性或 import 安全的必要前提。

## 6. Phase 3：实施移植

按从低层到高层推进。各层的文件布局、模板与 `setup.py` 具体改动点见 `kernel-integration-chain.md` §3–§5；本节只记移植特有的决策。

### A. Native 代码与依赖边界

1. 只把必需的 native 闭包搬进 `csrc/kernels/<domain>/`。
2. 新代码放 `namespace cache_dit::<domain>`，上游辅助名用匿名 namespace 留在内部。已迁移的 SVDQ 树保留上游的 `svdq::` namespace（它是上游派生代码），不要在无关任务里顺手改名。
3. 功能上必要的 dispatch 结构要保留（SVDQ 需要它那套继承式 launch dispatch），不要因为"看起来陌生"就重写。
4. 依赖策略必须显式决定：
   - vendored 进树
   - git submodule（目前只有 `csrc/third_party/spdlog`）
   - 依赖系统预装
5. 算子属于哪个家族就扩展哪个 binding TU；新算子家族要自己建 `<domain>/{<domain>.h,<domain>.cpp}`，不要借用别的家族的 TU。仓内 `PYBIND11_MODULE` 入口只有 `csrc/kernels/attn/attn.cpp`、`csrc/kernels/comm/comm_ops.cpp`、`csrc/kernels/svdq/pybind.cpp` 三个。

### B. 域 Python 模块

一个域的东西全部放进单个私有模块 `src/cache_dit/kernels/_<domain>.py`：loader、桥、注册、fake、公开封装。loader 与算子同模块，域内的失败形态、注册和公开 API 才不会各自漂移。每个域加载自己的扩展（`_attn.py` → `cache_dit._C_attn`、`_comm.py` → `cache_dit._C_comm`、`_svdquant.py` → `cache_dit._C_svdquant`）；只有当算子确实在兄弟扩展里时才借用兄弟的 loader。

模板与它强制遵守的规则：`kernel-integration-chain.md` §4.1–§4.5。

### C. 公开面

在 `src/cache_dit/kernels/__init__.py` 里显式 re-export，并加 `__all__` 条目：

```python
from ._comm import fp8_comm_per_token_quant
```

默认约定：

- 单层扁平命名空间，与域模块一一对应，不再加子包 re-export 层
- 签名可发现性重要时用显式函数，而不是 `partial(...)` 别名
- `__all__` 必须等于全部非下划线公开名，由 `test_ops_dispatch.py::test_public_api_matches_all` 校验
- 每个名字必须与其域模块里的是同一个对象，由 `test_public_api_is_reexported_from_domain_module` 校验

不要为了"集中管理"新建 `kernels/ops.py` 或注册表模块。那套集中层是被刻意删掉的，加回来正是政策禁止的回归。

### D. 上层模块与状态适配

如果移植同时引入模块抽象（例如量化 `nn.Module`）：

1. 模块期望的 state key 保持稳定
2. 把上游的原始导出 key 显式适配成 cache-dit 的模块 key
3. cache-dit 已有更合适的约定时，别把上游命名漏进公开 API
4. 模块走公开封装，不要直接调 `_call_*` 或 `torch.ops`

## 7. Phase 4：分层验证

不要只靠一个测试。验证通常按这个顺序推进：

1. **import 安全** —— 没有可选扩展时，import cache-dit 不应崩
2. **API 表面** —— 公开名 re-export 是同一对象，`__all__` 一致
3. **底层 smoke** —— 算子在预期 dtype / device / shape 下能跑
4. **底层正确性** —— 与 dense 或参考实现对比
5. **模块正确性** —— 上层模块确实走了移植后的算子路径
6. **fake / compile 路径** —— 至少一个 `register_fake` 的 shape+dtype 断言，加一个 `fullgraph=True` 的 compile smoke
7. **往返或端到端** —— 有序列化、量化或 pipeline 集成时必须单独测
8. **边界** —— 不支持的几何、rank、对齐或构建条件要清晰失败

对应测试文件：

- `tests/kernels/test_ops_dispatch.py` —— re-export 身份、`__all__`、已删 dispatch 的回归守卫
- `tests/kernels/test_cuda_fp8_comm_ops.py` —— FP8 comm 正确性、fake 路径、空输入、compile smoke
- `tests/kernels/test_cuda_merge_attn_states.py` —— attention 状态合并正确性与 compile
- `tests/kernels/test_svdquant_runtime.py` —— SVDQ 运行期行为与 INT4 v1/v2 等价
- `tests/quantization/` —— 模块层与量化器行为

范围被刻意限制时要写明，例如：

- "INT4 CUDA 路径是本轮验收门禁。"
- "FP4 代码保留，但当前不由运行期正确性测试把关。"

不要把成熟度说得超过测试实际覆盖。

## 8. Phase 5：打包与文档

构建与打包不对，移植就没完成。`setup.py` 的 6 处改动与 4 个"族清单"同步点在 `kernel-integration-chain.md` §5.1、§5.3 只维护一处，照那张清单做。移植侧另外记几条：

1. `_missing_*_sources()` 会把漏登记变成清晰的构建错误；反过来，磁盘上有但清单里没有的源文件会**静默不参与编译**。
2. 没有扩展时默认安装与 import 必须安全：除了惰性 loader，任何地方都不要急切 import 扩展。
3. `BuildExtension` 只比源文件 mtime，所以改了 `csrc/kernels/attn/attn.h` 这类共享头之后要清 `build/`（`CACHE_DIT_CLEAN=1`），不要相信缓存的 `.o`。
4. wheel 内容预期变化时更新 `tools/inspect_release_wheel.py`，例如 SVDQuant wheel 里必须多一个 `.so`。
5. 新增开关或宿主要求要更新 `docs/user_guide/INSTALL.md`。
6. 只写现在真的能用的东西，不要在验收前就在 README 或用户文档里宣传未完成能力。

## 9. 反模式

### 不要加 Triton 或 CuTe DSL 算子

坏：把 Triton kernel 原样搬进来，或以"可移植性"为名复活 `kernels/triton/`。
好：把算法重写成 `csrc/kernels/<domain>/` 下的 CUDA kernel。

### 不要恢复后端分派层

坏：新增 `kernels/backend.py`、`KernelBackend` 枚举或 `_resolve_backend` 来在多份实现间切换。
好：单一 CUDA 路径 + 一个 `torch.library` schema + 扩展缺失时点名构建命令的 `RuntimeError`。

### 不要机械镜像上游布局

坏：因为一个算子用了上游两个文件，就把整棵子目录拷进 `csrc/`。
好：找出最小闭包，只搬这一组。

### 不要把一个域拆散在多个文件里

坏：一个模块注册算子、另一个模块加载扩展、第三个模块做封装。
好：一个 `_<domain>.py` 里同时放 loader、桥、注册、fake 与封装。

### 不要随手暴露内部控制旋钮

坏：因为开发时方便，就把 stage 数、launch 调参或仅迁移期使用的开关暴露给用户。
好：在内部封装层写死，直到出现真实产品需求。

### 不要在上游命名之外重复发明命名

坏：cache-dit 已有约定时，仍把上游原始辅助名或 state key 留在公开接口上。
好：适配成仓库的公开命名规则，原始名按需留在私有侧。

### 不要让可选扩展打断基础 import

坏：在顶层包 import 路径里急切 import 扩展。
好：延迟到真正需要该算子时再 import（loader 形态见 `kernel-integration-chain.md` §4.1）。

### 不要用一个 smoke test 宣称正确

坏：扩展编译通过就宣布移植完成。
好：证明 import 安全、API 表面、底层执行、底层正确性、上层模块行为与边界失败。

### 不要写本机绝对路径

坏：在 docstring、错误信息或 skill 引用里写 `/abs/path/to/workspace/...`。
好：`src/cache_dit/kernels/_comm.py`、`nunchaku/nunchaku/models/linear.py`、`deepcompressor/deepcompressor/calib/smooth.py`。

## 10. 参考案例：SVDQ / Nunchaku 移植

把它当流程示例，不要逐行复刻。

### 这次移植证明了什么

- native W4A4 闭包可以不整包引入上游就迁过来（`csrc/kernels/svdq/` 后来重组为 `core/`、`int4/`、`fp4/`，这些现在位于 `csrc/kernels/svdq/native/` 之下、与 `csrc/kernels/svdq/cute/` 并列；这是本地布局选择，上游 nunchaku 保持扁平的 `csrc/`）
- 公开算子封装要显式、与仓库约定对齐
- `torch.library` 的 schema 与实现要和域封装放在一起，统一在 `cache_dit_cuda_ops` 命名空间
- 可选扩展的 import 要延迟，加载错误要可查询
- 打包可能要靠 submodule 强制，而不是把依赖硬编码进树
- 测试分层要分别覆盖运行期、模块与端到端行为

### 后续那次重构改了什么

comm 与 attention-state-merge 两个算子原本是 Triton 与 CuTe DSL 实现的 CUDA 版移植。那次迁移是后续工作的模板：

- native 源文件分别落在 `csrc/kernels/comm/` 与 `csrc/kernels/attn/`
- 每族通过自己的 pybind TU 注册：FP8 comm 是 `cache_dit._C_comm`，attention 状态合并是 `cache_dit._C_attn`（拆分前两者共用 `_C_comm`）
- 它们的 Python 模块把注册与封装放在一起，取代旧的 `cuda/_ops_registery.py` + `ops.py` 组合
- Triton 与 CuTe DSL 两棵树、后端选择器及其测试被删除，现在 `tests/kernels/test_ops_dispatch.py` 防止它们回来

### 有用的 cache-dit 参考文件

- `src/cache_dit/kernels/__init__.py`
- `src/cache_dit/kernels/_attn.py`
- `src/cache_dit/kernels/_comm.py`
- `src/cache_dit/kernels/_svdquant.py`
- `src/cache_dit/quantization/svdquant/linear.py`
- `csrc/kernels/attn/attn.cpp`
- `csrc/kernels/attn/attn.h`
- `csrc/kernels/attn/merge_states.cu`
- `csrc/kernels/comm/comm_ops.cpp`
- `csrc/kernels/comm/comm_ops.h`
- `csrc/kernels/svdq/pybind.cpp`
- `tests/kernels/test_ops_dispatch.py`
- `tests/kernels/test_cuda_fp8_comm_ops.py`
- `tests/kernels/test_cuda_merge_attn_states.py`
- `tests/kernels/test_svdquant_runtime.py`
- `tests/quantization/test_svdquant_quantizer.py`
- `setup.py`
- `tools/inspect_release_wheel.py`

### 有用的外部参考案例

- `nunchaku/nunchaku/models/linear.py`
- `nunchaku/nunchaku/ops/gemm.py`
- `nunchaku/nunchaku/ops/quantize.py`
- `deepcompressor/deepcompressor/backend/nunchaku/utils.py`
- `deepcompressor/deepcompressor/calib/smooth.py`
- `deepcompressor/deepcompressor/calib/lowrank.py`

### 哪些是那次案例特有的

这些细节对 SVDQ 很重要，但不是普遍移植规则：

- `svdq_*` 命名约定
- W4A4 的 INT4 与 FP4 在范围与验收上的拆分
- 具体的几何与 rank 约束
- 量化模块的状态适配规则
- `spdlog` 用 submodule 的选择

如果你的算子族不同，保留流程，但把决策重新评估一遍。

## 11. 执行顺序与完成判据

真实移植建议按这个顺序：

1. 确认目标是 CUDA；源头是 Triton 或 CuTe DSL 就先规划重写
2. 概括目标算子与当前范围边界
3. 列出源文件与 cache-dit 侧的宿主模块
4. 确定最小可用闭包
5. 冻结公开 API 与命名策略
6. 迁 native 代码并扩展 pybind TU
7. 在一个域模块里补齐 loader、桥、注册、fake、封装
8. 从 `kernels/__init__.py` 导出
9. 接 `setup.py` 的源清单与构建门禁
10. 需要时加上或适配上层模块代码
11. 按 kernel、模块、端到端三层验证
12. 只把已验证的范围写进文档

以下全部为真才算完成：

- 算子是 CUDA 的，改动里没有任何 Triton 或 CuTe DSL 成分
- 公开 API 是有意设计的、与仓库约定一致
- 可选扩展缺失时的行为是安全的
- 依赖策略已明确
- 每个新增 native 源文件都出现在 `setup.py` 的源清单里
- 测试在正确的层级证明了正确性（kernel 层与模块层按需）
- 不支持的场景会清晰失败
- 文档与已验证的现实一致
