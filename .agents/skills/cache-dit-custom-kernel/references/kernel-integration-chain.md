# Cache-DiT CUDA 算子接入链路（通用）

> 日期：2026-11-05。路径均相对 cache-dit 仓库根目录。
> 本文覆盖"从零接入一个新算子/新算子族"的完整链路；从上游仓库搬算子的判断流程见 `operator-migration.md`，SVDQ 的实现、性能与数学见同目录其它文档。

## 1. 链路总览：8 个接触点

一次调用真实经过的层（以 `fp8_comm_per_token_quant` 为例）：

```text
调用方 distributed/communicate/primitives.py:802
  -> torch.ops.cache_dit_cuda_ops.fp8_comm_per_token_quant   dispatch：FakeTensor / compile / device key
  -> kernels/comm.py:124  @torch.library.impl(..., "CUDA")   Python impl：元数据校验 + 建输出缓冲
  -> kernels/comm.py:77   _call_fp8_comm_per_token_quant     桥：out-param 适配
  -> _C_ops.ops.fp8_comm_per_token_quant                      pybind submodule（单扩展，四族共用）
  -> csrc/kernels/comm/comm.cpp -> comm/fp8_comm.cu           host 校验 + kernel launch
```

新增算子/新族要动的接触点（缺任一处都表现为"静默失效"或裸 `AttributeError`）：

| # | 接触点 | 缺了会怎样 |
|---|---|---|
| 1 | `csrc/kernels/<族>/*.cu`（可含 `.cuh`） | 没有实现 |
| 2 | 声明头 `<族>/<族>.h`（§3.1） | 链接失败 |
| 3 | registrar TU `<族>/<族>.cpp` 里 `register_<族>_ops`（§3.2） | `AttributeError: module has no attribute '<op>'` |
| 4 | `setup.py` 的 `_required_cuda_sources()` / include_dirs（§5.1） | 不编译、漏编 |
| 5 | `src/cache_dit/kernels/<族>.py` 六段（§4.1） | `torch.ops` 未注册 |
| 6 | `src/cache_dit/kernels/__init__.py` re-export + `__all__` | 内部调用拿不到名字 |
| 7 | 内部调用点：可用性门禁 + 错误信息（§6） | 用户看到与根因无关的报错 |
| 8 | `tests/kernels/` 用例 + `test_ops_dispatch.py` 参数表（§7） | 无回归 |

**两套并行注册表（最容易找错地方）**：

- `cache_dit_cuda_ops` 命名空间只存在于 **Python**：由 `_<族>.py` 在 import 时调用 `torch.library.define` 创建。`csrc/` 里 **没有任何 `TORCH_LIBRARY` / `TORCH_LIBRARY_IMPL`**，不要去找。
- C++ 侧只有一个 pybind 子模块 `ops`，由**唯一的** `csrc/kernels/ops.cpp` 建立；各族只提供 `register_<族>_ops(py::module_&)` 往里挂。
- `_call_*` 就是这两套之间的桥：C++ 一律 out-param，schema 默认函数式。

**import 时机**：`import cache_dit` **会**把 `cache_dit.kernels` 拉起来（经 `attention/__init__.py` 的依赖链），所以 `.so` 的加载发生在包导入期。但 loader 内部 `except Exception` 吞掉并缓存错误，**缺 `.so` 不会让 import 失败**，只会在真正调用算子时抛 `RuntimeError`；import 期唯一会硬失败的情况是重复注册（§4.5 第 3 条）。

## 2. 四个算子族的现状（符号级盘点）

**只有一个扩展**：`cache_dit._C_ops`，入口 TU 是 `csrc/kernels/ops.cpp`（唯一 `PYBIND11_MODULE`），四个族各自提供 registrar。

| 族 | registrar TU | 声明头 | C++ namespace | Python 模块 | schema 数 | 内部调用点 | 偏离模板处 |
|---|---|---|---|---|---|---|---|
| comm | `comm/comm.cpp` | `comm/comm.h` | `cache_dit::comm` | `comm.py` | 4 | `distributed/communicate/primitives.py` | impl 显式 dtype `ValueError`；fake 不复刻 reshape |
| attn | `attn/attn.cpp` | `attn/attn.h` | `cache_dit::attn` | `attn.py` | 1 | `distributed/parallel/ring.py` | impl 无 dtype 校验；fake 复刻 view 链 |
| svdq | `svdq/svdq.cpp` | `svdq/svdq.h`（纯声明） | `cache_dit::svdq`（dispatcher）/ `svdq::native_kernels`、`svdq::cute_kernels`（实现，仍在全局 `svdq`） | `svdquant.py` | 2（`gemm_w4a4` / `quantize_w4a4`） | `quantization/svdquant/`（`linear.py`、`fused.py` 等） | 形状/通道校验在 Python，C++ 侧无 `TORCH_CHECK`；`runtime.h` 的 `RuntimeParams`/`LoraLayout` 仍属全局 `svdq`，`svdq.cpp` 里用 `using ::svdq::...` 引入 |
| models | `models/models.cpp`（族级 registrar，各模型 `X.cpp` 只注册） | `models/models.h` + `models/X.h` | `cache_dit::models` | `models/X.py` | 每模型自定（boogu: 3） | `models/pipelines/X/fused_ops.py`（模型侧 wrapper） | 多一层模型侧 wrapper + 逐位等价 eager 回退 |

**文件名契约统一为 `<族>/<族>.h` + `<族>/<族>.cpp`**（`comm.h`/`comm.cpp`、`attn.h`/`attn.cpp`、`models.h`/`models.cpp`、`svdq.h`/`svdq.cpp`）：头文件只声明、`.cpp` 里给定义并挂 registrar。**定义绝不能放头文件**——`ops.cpp` 会 include 四个族头，非 inline 定义会 `multiple definition` 链接失败（2026-11-11 实测）。

`models` 是**模型级**族：文件名契约是 `<model>.h` / `<model>.cpp` / `<model>_<op>.cu`，`models.cpp` 是族级 registrar，各模型只提供 `register_<model>_ops(py::module_&)`；新增模型不加 extension、不改 `setup.py`（源清单在 `_required_models_sources()` 里显式列出）。因此它比框架级三族多一层 `models/pipelines/<model>/fused_ops.py`：算子是否启用、以及未启用时的等价表达式，都由模型侧决定。命名与硬规则见 `SKILL.md` §9。

arch 门槛只剩一个：`_is_supported_cuda_sm_target`（`>=sm89` 且排除 `sm90`），是所有族的交集，Hopper 上整个扩展不编译。

盘点表刻意只列符号级信息：算子名逐个列举、行号、mtime 都会腐化，查具体算子请直接看 pybind TU。

## 3. C++ 层

### 3.1 声明头

```cpp
// csrc/kernels/<族>/<族>.h
#pragma once

#include <ATen/ATen.h>

namespace cache_dit::<族> {

void <op>(at::Tensor x, at::Tensor out);

} // namespace cache_dit::<族>
```

SVDQ 的签名是唯一例外：`svdq.h` **只声明**、`svdq.cpp` 里定义，形参用 `std::optional<torch::Tensor>` + `TorchOpContext`，因为要在 C++ 侧按 dtype/fp4/lora_layout 分派。**头里绝不能放非 inline 定义**——`ops.cpp` 与本族 registrar 都会 include 这个头，会链接期 `multiple definition`（2026-11-11 实测）。只在确实需要 C++ 侧分派时才照抄这种签名写法。

### 3.2 pybind TU

```cpp
// csrc/kernels/<族>/<族>.h
#pragma once
#include <ATen/ATen.h>

namespace pybind11 { class module_; }  // 只在头里前置声明，别把 torch/extension.h 拉进 device 侧

namespace cache_dit::<族> {
void <op>(at::Tensor x, at::Tensor out);
void register_<族>_ops(pybind11::module_& ops);   // registrar 声明
}

// csrc/kernels/<族>/<族>.cpp
#include <torch/extension.h>
#include "<族>.h"

namespace cache_dit::<族> {
void register_<族>_ops(pybind11::module_& ops) { ops.def("<op>", &<op>); }
}
```

```cpp
// csrc/kernels/ops.cpp —— 全仓唯一 PYBIND11_MODULE
#include <torch/extension.h>
#include "attn/attn.h"
#include "comm/comm.h"
#include "models/models.h"
#include "svdq/svdq.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  auto ops = module.def_submodule("ops");
  cache_dit::attn::register_attn_ops(ops);
  cache_dit::comm::register_comm_ops(ops);
  cache_dit::models::register_models_ops(ops);
  cache_dit::svdq::register_svdq_ops(ops);
}
```

- 模块名由 `CUDAExtension(name="cache_dit._C_ops")` 决定，`TORCH_EXTENSION_NAME` 自动匹配，所以 `.so` 里只有 `PyInit__C_ops`。
- 只有 `ops` 一个子模块。`def_submodule()` 返回**临时对象**，绑定到 `module_&` 形参编译不过——要挂子模块时先 `auto utils = module.def_submodule("...");` 再用（现状没有辅助子模块：`set_log_level` 已随无调用者删除）。
- `py::arg("...")` 可选。SVDQ 的 `gemm_w4a4` 给了命名参数，comm/attn 没给；Python 侧 `_call_*` 建议全位置传参，避免与 `py::arg` 名字耦合。

### 3.3 kernel TU 与校验分层

C++ 入口**一律 `void` + out-param**（三族 100%）。

校验现状并不统一，新增算子按前者写：

- **comm / attn：`TORCH_CHECK` 是权威**——检查 `is_cuda()`、同 device、输入/输出 dtype、`is_contiguous()`、rank 与形状、`head_size` 之类整除约束（见 `comm/fp8_comm.cu`、`attn/merge_states.cu` 的入口段）。Python impl 只补能在 FakeTensor 下复现的元数据检查（dtype/rank），因为它比 C++ 早失败、报错更贴近用户。
- **svdq：两侧都几乎没有校验**——`csrc/kernels/svdq/` 里 `TORCH_CHECK` 命中数为 0，`svdquant.py` 只有形状/通道与输出 dtype 的检查，没有输入 dtype 校验。这是迁移遗留，**不要当模板**。

`_call_*` 与 `ops.*` 是外部可绕过的后门，只靠 Python 层校验会漏，所以新算子要把硬约束放进 C++。

### 3.4 schema 的决策轴（不是"要不要 out-param"）

C++ 侧已是 out-param，真正要定的是 schema 形状：

- **单输出还是多输出**：comm 全单输出；attn 双输出（`-> (Tensor out, Tensor lse)`）；svdq 的 `quantize_w4a4` 是三输出。
- **输出 dtype/形状由谁决定**：FP8 输出必须由 Python 建 `torch.empty(..., dtype=torch.float8_e4m3fn)`；svdq 的 `oscales` dtype 与形状随 `fp4` 开关变化（fp4 → `float8_e4m3fn` + `channels/16`，否则 `input.dtype` + `channels/64`）。凡是输出 dtype/形状依赖 Python 侧参数的算子，out-param 就是唯一可行解。

**默认把 out-param 藏在 `_call_*` 里，schema 只暴露函数式返回**——这样 functionalization 看到的是普通"返回新张量"的算子，不需要 `mutates_args`。

例外是 svdq 的 `svdq_gemm_w4a4`：融合 MLP 路径要求把 `qout` / `oscales` / `lora_act_out` 这些 in-place 输出缓冲列进 schema，因此该调用形态**不能进图**，上层 `quantization/svdquant/fused.py` 的 `fused_gelu_mlp` / `fused_gelu_proj` 必须标 `@torch.compiler.disable`。`out` 通常由 impl 内部分配并返回，但 `write_dense=False` 时传 `None` 让 kernel 跳过 dense store（融合 MLP 只吃 qout），此时返回 0 元素占位张量——占位是刻意设计的哨兵，不是缺陷。新增算子不要复制这种形态，除非确实需要"一块大 kernel 同时写多个输出"。

## 4. Python 层

### 4.1 规范模板（新算子照抄这一段）

```python
# 共用 loader：src/cache_dit/kernels/cuda_ext.py（四族唯一一份，不要复制）
from ._cuda_ext import _get_required_ops_module

def _call_<op>(...) -> None:                             # 桥：out-param 适配 + 签名归一化

torch.library.define("cache_dit_cuda_ops::<op>", "(Tensor x) -> Tensor")

@torch.library.impl("cache_dit_cuda_ops::<op>", "CUDA")
def _<op>(x: torch.Tensor) -> torch.Tensor:              # 元数据校验 + 建输出 + 调 _call_<op>

@torch.library.register_fake("cache_dit_cuda_ops::<op>")
def _fake_<op>(x: torch.Tensor) -> torch.Tensor:         # 形状/dtype 与 impl 必须逐位一致

def <op>(x: torch.Tensor) -> torch.Tensor:               # 公开包装：docstring + 调 torch.ops
  return torch.ops.cache_dit_cuda_ops.<op>(x)

__all__ = ["<op>", ...]
```

`cuda_ext.py` 提供 `cuda_ext_is_available()` / `cuda_ext_get_load_error()` / `_get_required_ops_module()`，其中 `_EXTENSION_MODULE_NAME = "cache_dit._C_ops"`。**不要再写 per-family loader 或 `xxx_extension_is_available`**：那 8 个名字在 2026-11-11 合并时全部删除，`tests/kernels/test_ops_dispatch.py::test_per_family_availability_getters_removed` 守着它们不再回来。

### 4.2 三族现状偏离（不要把现状当模板）

| 维度 | comm | attn | svdq |
|---|---|---|---|
| required getter | 共用 `_cuda_ext._get_required_ops_module` | 同左 | 同左 |
| 非 Tensor 算子 | 无 | 无 | 无（`set_log_level` 已删除） |
| impl 内输入 dtype 校验 | 有（`ValueError`） | 无 | 无（只校验形状/通道与输出 dtype） |
| C++ `TORCH_CHECK` | 有 | 有 | 无 |
| fake 写法 | 直接 `new_empty(最终形状)` | 复刻 impl 的 flatten/view 链 | 视算子而定 |
| `__all__` 是否收 `_call_*` | 收 | 不收 | 不收 |

### 4.3 `_call_*` 的职责边界

它只为"C++ out-param ↔ schema 函数式"而存在。可以承担**签名归一化**：dtype→整数编码、按 rank 补齐 scale 列表、`None`→默认值（见 `svdquant.py` 的 `_encode_svdq_output_dtype`、`_normalize_svdq_lora_scales`）。**不可以**承担输出形状/dtype 的推导——那属于 impl，否则 fake 必然漂移。

### 4.4 fake 的两条规则（条件式）

- 输出由 `view`/`reshape` 从另一块缓冲派生 → **必须复刻同一条链**，否则 stride 语义不一致（`attn.py` 的 fake 里有注释说明这一点）。
- 逐元素同构的重排（例如尾部多几列 scale）→ `x.new_empty(最终形状, dtype=...)` 即可，不必复刻（`comm.py` 就是这么写的）。

两条硬要求：fake 内不得出现数据依赖分支；fake 的 shape/dtype 必须与 impl 完全一致——**不一致时 eager 全绿、`torch.compile` 才炸**。

### 4.5 schema 的三条陷阱（torch 2.11.0 实测）

1. **kwarg 只认 schema 形参名**：`torch.ops.ns.op(a=..., b=...)` 按 schema 名解析；用 impl 的形参名当 kwarg 直接 `RuntimeError`。
2. **位置参数按 schema 顺序转发，impl 形参顺序写反不报错、静默算错**：把 `(a, b) -> a - b` 的函数注册成 `(Tensor a, Tensor b)`，调用 `op(10, 1)` 得到的是 `1 - 10`。schema、pybind、impl 三处声明**没有任何自动守卫**，只能靠人工核对参数个数/顺序 + 家族 numerics 测试兜底。
3. **`define` / `impl` 不幂等**：同名重复 `define` 报 `Tried to register an operator ... multiple times`，重复 `impl(..., "CUDA")` 报 `This is not allowed since there's already a kernel registered`。推论：不要 `importlib.reload` 这些模块，也不要让同一模块经两条路径加载（editable install 叠加源码路径、vendored 副本都会在 import 期炸）。

## 5. 注册与构建

### 5.1 `setup.py` 的改动（只有一个扩展）

| # | 位置 | 作用 |
|---|---|---|
| 1 | `_required_<族>_sources()` | 显式族源清单（`.cpp` + 每个 `.cu`）；`comm.cpp`、`attn.cpp`、`models.cpp`、`svdq.cpp` 都在这里 |
| 2 | `_required_cuda_sources()` | `csrc/kernels/ops.cpp` + 四个族清单的并集，就是 `_C_ops` 的 `sources` |
| 3 | `_missing_cuda_sources()` | 构建前整体校验清单里的文件都在（per-family 的 `_missing_*` 已删除，只有 `tests/kernels/test_models_source_manifest.py` 还在用 `_missing_models_sources`） |
| 4 | `_get_cuda_extension()` | 唯一的 `CUDAExtension(name="cache_dit._C_ops", ...)`：`depends=` 扫全 `csrc/kernels/**/*.{h,cuh}`，`include_dirs` 给 `csrc/kernels` + 四个族目录 + spdlog + CUTLASS |
| 5 | `_is_supported_cuda_sm_target()` | 唯一的 arch 谓词：`>=sm89` 且排除 `sm90` |
| 6 | `IS_CU13_RELEASE` 分支 | cu13 发布件缺扩展要 `raise RuntimeError`，普通构建走 `_warn_skipped` |

**新增族不要再新建 extension**：加一份 `_required_<族>_sources()`、把它并进 `_required_cuda_sources()`、把族目录加进 `include_dirs`、在 `csrc/kernels/ops.cpp` 里多调一行 registrar 即可。`_DBUILD_CACHE_DIT_<族>=1` 宏是**惰性标签**（源码零引用）。

`models` 族的特殊处只剩一条：新增**模型**（新 `<model>.h/.cpp/*.cu`）必须同时加进 `_required_models_sources()`（显式列表，不是 glob），否则永远不参与编译。

### 5.2 arch 谓词与 skip 语义

- `_is_supported_cuda_sm_target`：`>=sm89` 且排除 `sm90`（四族门槛的交集；Hopper 上整个扩展不编译）。
- `_resolve_sm_targets(scope, is_supported)` 只负责过滤，返回空列表时由调用方决定 skip 还是 raise。
- `_warn_skipped` 用 `warnings.catch_warnings()` + `simplefilter("always", CacheDiTSkipWarning)` 包住：即使调用方开了 `-W error`，"skip"也不会变成构建失败（这是设计意图，别改成裸 `warnings.warn`）。

### 5.3 单扩展下的"族清单"同步点（不报错、易漏）

- `_required_cuda_sources()` 是 `sources` 的唯一来源：新族只加 per-family 清单、不并进这个函数，就**不会被编译**。
- `_get_cuda_extension()` 的 `include_dirs`：新族目录没加就不会被 `#include "<族>/<族>.h"` 找到。
- `csrc/kernels/ops.cpp`：新族 registrar 没挂进 `PYBIND11_MODULE`，`.so` 里就没有该族的算子（运行时裸 `AttributeError`）。
- `tools/inspect_release_wheel.py` 的 `--so svdquant` 分支：现在只检查单一 `_C_ops`；`--so` 的两个取值 `none|svdquant` 是历史命名，被 `tools/build_releases.sh` 调用，**不要改名**。
- `docs/user_guide/INSTALL.md` 里"builds the single `cache_dit._C_ops` extension"那一段的族列举与 arch 门槛。
- `tests/kernels/test_cuda_ext_manifest.py`：CPU-only 双向比对 `_required_cuda_sources()` 与 `csrc/kernels/**/*.{cu,cpp}`，新族漏登记会在这里失败（这是唯一自动护栏）。

### 5.4 两个构建细节

- `-DBUILD_CACHE_DIT_{SVDQUANT,COMM,ATTN,MODELS}=1` **是空宏**：`csrc/` 里零引用，不要描述成条件编译开关，也不要以为删掉它会变行为（保留它们只是让编译命令行还能看出链接了哪些族）。
- `--generate-line-info` 只在 `CACHE_DIT_BUILD_WHEELS` 为 `0` 或未设置（即非 wheel 构建）时加。借它可以把源码绝对路径写进 `.so`，用 `strings <so> | grep <路径片段>` 反查"这个产物到底是不是按当前源码重编的"；wheel 构建没有这个能力，改用 `nm -D` / `inspect_release_wheel.py`。

### 5.5 新增族的最小编译与冒烟

```bash
CACHE_DIT_BUILD_CUDA=1 CACHE_DIT_CUDA_ARCH_LIST=120a \
  bash tools/build_fast.sh install -e . --no-build-isolation --no-deps

python -c "import cache_dit._C_ops as m; print(m.__file__, sorted(dir(m.ops)))"
python -c "from cache_dit.kernels import cuda_ext_is_available as f; print(f())"
python -c "import cache_dit.kernels as k; print(k.<op>)"
```

`kernels/_init__.py` re-export 的是算子包装 + 唯一的 `cuda_ext_is_available` / `cuda_ext_get_load_error`，冒烟命令直接可用。

`build_ext` 比较源文件 mtime，`depends=` 只影响"头文件也参与 mtime 比较"。**现在 `_get_cuda_extension()` 对全 `csrc/kernels/**/*.{h,cuh}` 声明了 `depends=`**，改任何族的头都会重编（这是合并的附带收益，见 §8）。

## 6. 内部调用接入

- **唯一有效的可用性判定**：`cuda_ext_is_available()`，或捕获 `_get_required_ops_module()` 抛出的 `RuntimeError`。单扩展下"某族可用"这种判定不存在——要么全有，要么全无。
- **`except ImportError` 是历史写法，对"扩展没编译"无效**：`_load_extension_module()` 内部已经 `except Exception` 吞掉并缓存，`torch.library.define` 也不需要扩展，所以 `from ...kernels import <op>` 永远成功。现状 `communicate/primitives.py:16-33` 与 `parallel/ring.py:5-9,76` 的降级分支只在"模块本身不存在（旧版本 cache-dit）"时才可能命中；缺 `.so` 时的真实行为是**调用即抛带构建命令的 RuntimeError**——Ring Attention 在无 `_C_ops` 时是硬失败，不是优雅降级。
- 错误信息必须带构建命令与 arch 门槛，照抄 `_get_required_ops_module()` 的文案；否则用户只会看到 dispatch 报错或裸 `AttributeError`。
- **autograd**：三族都只注册了 `CUDA` key，**没有 Autograd key**。这些算子是推理专用：`requires_grad=True` 的输入走 `autograd_not_implemented` fallback（torch 2.11 实测为 `UserWarning`，并注明该行为未来会变成 error）。要可微必须显式注册 Autograd 实现（或包一层 `autograd.Function`），不要指望"能跑通就是对的"。
- **torch.compile**：注册了 fake + fake 无数据依赖分支 + 公开包装里不出现 `.item()`/值比较，即可 `fullgraph=True`。模板见 `tests/kernels/test_cuda_merge_attn_states.py` 与 `tests/kernels/test_cuda_fp8_comm_ops.py::test_fp8_comm_compile_smoke`。
- **模型级族（`models`）多一层 wrapper**：调用点不在框架代码里，而在 `models/pipelines/<model>/fused_ops.py`。它负责三件事——前置条件（必须覆盖 C++ 的 `TORCH_CHECK`）、逐位等价的 eager 回退、以及开关。开关统一为 `CACHE_DIT_FUSED_OPS`（解析点在 `models/pipelines/env.py`，三态；由 `install_fused_modules()` 安装 fused transformer 时打开，`boogu_image` 这种无独立安装步的 vendored 模型则以构造其 transformer 为激活点），不是模型专属变量。框架级三族没有这层义务，所以别把三族的写法直接套到模型侧。

## 7. 测试矩阵（含现有用例的真实强度）

| 目标 | 用例 | 强度与局限 |
|---|---|---|
| 包内 re-export、`__all__` 自洽 | `tests/kernels/test_ops_dispatch.py` | 参数表**手工维护**，新算子不加条目等于没测；断言只保证 `kernels.__all__` 与 `vars(kernels)` 自洽（`!= "kernels"` 豁免），import 与 `__all__` 两边都漏时依然全绿 |
| fake 注册 | `test_cuda_fp8_comm_ops.py::test_fp8_comm_fake_registration_shapes`、`test_cuda_merge_attn_states.py::test_fused_merge_attn_states_fake_registration` | `FakeTensorMode` 构图即可，CPU 能跑 |
| 数值正确 | `test_cuda_fp8_comm_ops.py`（packing 参考）、`test_cuda_merge_attn_states.py`（float32 参考实现逐元素比）、`test_svdquant_runtime.py`（INT4 smoke + 融合 GELU epilogue 对 `gelu(out)` 的等价护栏） | 需 CUDA + 扩展 |
| 边界与空输入 | `test_cuda_fp8_comm_ops.py`（未定义布局拒绝、empty input） | 需 CUDA + 扩展 |
| 无扩展时的行为 | 各测试文件里的 `_require_*()` 守卫（现状 2 个文件有） | 只 `pytest.skip`，不断言错误文案 |
| 源清单与磁盘一致 | `tests/kernels/test_cuda_ext_manifest.py`（`_C_ops` 并集）+ `test_svdquant_source_manifest.py`（svdq 族）+ `test_models_source_manifest.py`（models 族） | 全部 CPU-only、双向比对；新族漏登记在这里失败 |
| 统一可用性 API 未被拆回 per-family | `tests/kernels/test_ops_dispatch.py::test_per_family_availability_getters_removed`、`::test_single_extension_module_name` | 只防名字回归，不校验 `.so` 内容 |
| compile 可用 | `test_cuda_fp8_comm_ops.py::test_fp8_comm_compile_smoke`、`test_cuda_merge_attn_states.py` 的 `fullgraph=True` | 单算子 smoke |

**新族照抄的测试模板是 `tests/kernels/test_cuda_fp8_comm_ops.py`**：fake 形状断言、参考实现对比、strided/empty 边界、compile smoke 四类齐全。新增算子至少补：`test_ops_dispatch.py` 参数表条目、fake 用例、参考实现对比、compile smoke；改源文件名/新增 `.cu` 时同步 §5.1 的源清单并跑 `test_cuda_ext_manifest.py`（`_C_ops` 全量）+ 对应族的 manifest 护栏。`tests/kernels/` 不在任何 CI 里（`.github/workflows/cpu-tests.yml` 只跑 4 个 `api/test_*.py`，且 workflows 不许改），只能本地执行。

## 8. 坑清单（按"不报错"的危害排序）

1. **函数定义不要放族头**：`ops.cpp` 会 include 四个族的头，头里的非 inline 定义会在链接期 `multiple definition`（2026-11-11 合并 `svdq.h` 时实测，`collect2: error: ld returned 1 exit status`）。头只放声明 + `inline` 小工具。
2. **schema 参数顺序写错 → 静默算错**（§4.5 第 2 条）。
3. **陈旧 `.so` 缺新算子 → 裸 `AttributeError: ... has no attribute '<op>'`**，没有任何"请重建"提示。合并后更危险：旧 `_C_comm.so` / `_C_attn.so` 之类残留会和 `_C_ops.so` 并存，测试可能读到旧二进制。首次构建后确认 `src/cache_dit/` 下只有 `_C_ops*.so`。
4. **源清单漏登记**：清单有、磁盘缺 → 构建期 `RuntimeError`（好情况）；磁盘有、清单无 → 该文件根本不参与编译，**静默**。`test_cuda_ext_manifest.py` 是唯一自动护栏。
5. **`-DBUILD_CACHE_DIT_*` 是空宏**，别当开关。
6. **fake 与 impl 形状/dtype 不一致** → eager 绿、compile 炸。
7. **`define`/`impl` 不幂等** → reload 或双路径加载在 import 期炸。
8. **单扩展下"某族没编译"不存在**：一族源清单漏登记不会只丢那一族，而是整份清单一起编（漏的那个族运行时才炸）。
9. **无 autograd kernel** → 梯度静默错误，只有告警。
10. **`CACHE_DIT_BUILD_WHEELS=1` 的产物没有 line-info** → §5.4 的 `strings` 反查法失效。
11. **`py::` 命名空间遮蔽**：`cache_dit::svdq` 里写 `svdq::cute_kernels::...` 会解析到 `cache_dit::svdq` 自身（`cache_dit` 的成员名就是 `svdq`），必须写成 `::svdq::cute_kernels::...`。
12. **`def_submodule()` 返回临时对象**：`register_x(module.def_submodule("y"))` 编译不过（`cannot bind non-const lvalue reference`），先存局部变量。

## 9. 与其它 skill 的分工

- **从上游仓库移植算子**（nunchaku / deepcompressor / flashinfer 等）走同目录的 `operator-migration.md`。
- **NCU 驱动的迭代优化**走 `cuda-auto-tune`。
- **CUDA/PTX/架构细节**走 `cuda-cpp-kernel`；**CUTLASS/CuTe C++ 模板**走 `cutlass-cpp-kernel`；**CuTe DSL Python**（`@cute.kernel`）的 kernel 编写走 `cute-dsl-kernel`。本仓当前只有 CUDA/CuTe C++ 算子；将来若引入 CuTe DSL 算子，其接入、注册与构建仍按本文 §3–§5 走一条完整链路，Triton 则始终不支持。
