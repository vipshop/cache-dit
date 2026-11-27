# 接入链路：每个接触点的真实语义

所有路径相对 `cache-dit/`。行号会漂，用符号名 `file::symbol` 定位。

## 1. `_AttnBackend` 枚举

`src/cache_dit/attention/backends/register.py::_AttnBackend`，`(str, Enum)`，因此 `_AttnBackend.FOO == "foo"`。

```python
class _AttnBackend(str, Enum):
  """Cache-DiT owned attention backend identifiers."""

  NATIVE = "native"
  FLASH = "flash"
  FLASH_VARLEN = "flash_varlen"
  SAGE = "sage"
  SAGE3 = "sage3"
  SOL_ATTN = "sol_attn"
  _FLASH_3 = "_flash_3"
  _SDPA_CUDNN = "_sdpa_cudnn"
  FFPA = "ffpa"
  FFPA_FP8 = "ffpa_fp8"
  FFPA_FP4 = "ffpa_fp4"
  FFPA_FP8_PER_BLOCK = "ffpa_fp8_per_block"
  _NATIVE_NPU = "_native_npu"
  _NPU_FIA = "_npu_fia"
  _MINDIESD_LASER = "_mindiesd_laser"
```

要点：

- 成员**值**就是用户看到的字符串（`--attn`、`ParallelismConfig.attention_backend`、`processor._attention_backend`），改名等于破坏兼容。
- 下划线前缀成员（`_flash_3`）仍可正常访问；它表达"内部/特殊用途"，且能被 CLI `choices` 直接列出来。
- 默认 active backend 由 `_default_active_backend()` 从 `CACHE_DIT_ATTN_BACKEND` 读取，非法值只 warning 后退回 `native`（**注意：这里就是一处静默回退**，报错能力有限，别依赖它发现拼写错误）。

## 2. `_AttnBackendRegistry.register`

```python
@_AttnBackendRegistry.register(
  _AttnBackend.MY_BACKEND,
  constraints=[],                  # 可选检查函数，仅 CACHE_DIT_ATTN_CHECKS=1 时执行
  supports_context_parallel=True,  # 决定 _context_parallel_attention 是否允许
  bridge_to_diffusers=True,        # 是否注入 diffusers 注册表（默认 True）
)
def _my_attention(query, key, value, ...): ...
```

装饰器做四件事（全部在注册时快照，运行期不再重算）：

1. `_backends[name] = func`
2. `_constraints[name] = constraints or []`
3. `_supported_arg_names[name] = set(inspect.signature(func).parameters)` ← **dispatcher 的唯一过滤依据**
4. `supports_context_parallel` → `_supports_context_parallel` 集合；`bridge_to_diffusers` → `_bridge_to_diffusers` 集合

推论：

- 函数**签名就是契约**。少写 `attn_mask`，mask 会在 dispatch 层被丢掉且只打一条 warning。
- 事后 `functools.wraps` 替换函数、或 monkeypatch 换实现时，签名集合不会跟着变（测试里 monkeypatch 要同时 `setitem` `_supported_arg_names`，先例见 `tests/parallelism/test_attention_dispatch.py`）。
- `constraints` 是"可选加固"，不是主门。主门写在调用体里。
- 注册时机是 **import 时**。所以 `backends/__init__.py` 与 `attention/__init__.py` 两处 import 必须同步；后者是显式列表：

```python
from .backends import native, cudnn, sage, sage3, sol_attn, flash, flash_varlen, ffpa, npu
```

## 3. `_dispatch_attention_fn`

签名（节选）：`(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, attention_kwargs=None, cp_gqa_strategy=None, *, backend=None, cp_config=None, parallel_config=None)`。

行为顺序：

1. `cp_config` / `parallel_config` 归一化（`_resolve_cp_config`，两者都给且不同会 warning 并优先 `_cp_config`）。
2. `backend=None` → 取 `_active_backend`；显式给 → `normalize_backend` + `ensure_backend_registered`（失败会尝试 diffusers proxy），未注册则 `ValueError`。
3. 组装 kwargs：`query/key/value`、`attn_mask`、`dropout_p`、`is_causal`、`scale`、`**attention_kwargs`、`cp_gqa_strategy`、`_cp_config`；**`enable_gqa` 仅在 `_supports_enable_gqa()`（torch>=2.5）时加入**。
4. 若 `CACHE_DIT_ATTN_CHECKS=1`：对不被签名接受的 kwargs 打 warning，并逐个跑 `constraints`。
5. 按 `_supported_arg_names` 过滤后调用 `backend_fn(**kwargs)`。

这套设计意味着"多给参数"是安全的（被过滤），"少声明参数"是危险的（语义被扔）。所以**每个 backend 都声明完整 11 个参数**，不支持的显式 raise。

## 4. `_context_parallel_attention`

```python
def _context_parallel_attention(
  query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
  scale=None, enable_gqa=False, return_lse=False, cp_gqa_strategy=None,
  *, forward_op, backward_op, _cp_config=None,
):
```

前置校验（都在拿不到 `_cp_config` / 语义非法时直接 raise）：

- `attn_mask is not None` → 只允许 allowlist 内的 forward op 名：`_native_attention_forward_op`、`_cudnn_attention_forward_op`、`_flash_varlen_attention_forward_op`、`_npu_attention_forward_op`、`_npu_fused_infer_attention_forward_op`。**按 `forward_op.__name__` 字符串匹配**，新 backend 想支持 mask+CP 必须改这个列表，否则永远进不去。
- `is_causal` → 一律拒绝（templated attention 尚不支持）。
- `enable_gqa` 且 `cp_gqa_strategy is None` → 拒绝；`cp_gqa_strategy` 只允许 `None` / `replicate_kv_sequence` / `group_aligned_flash_varlen`，且只有纯 Ulysses 能带策略。
- 按 `ring_degree` / `ulysses_degree` 分派到 `RingAttention` / `USPAttention` / `UlyssesAttention`，三者都只做通信与合并，真正的单卡计算是你的 `forward_op`。

## 5. CP forward/backward op 契约

```python
def _my_attention_forward_op(
  ctx: torch.autograd.function.FunctionCtx,
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  _save_ctx: bool = True,
  _cp_config: Optional["_ContextParallelConfig"] = None,
):
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not supported under context parallelism for <backend>.")
  out = _my_attention_op(query, key, value, is_causal)
  if return_lse:
    return out, lse  # lse: [B, N, H]（Ring 会自己 unsqueeze 到 [B, N, H, 1]）
  return out
```

硬约束：

- Q/K/V 是**本 rank 的分片**，layout 仍是 NHD `[B, S_local, H, D]`。
- `_save_ctx=False` 时不要碰 `ctx`（Ring 明确传 `False`）。
- **Ring 每次都传 `return_lse=True`**（`distributed/parallel/ring.py`），所以返回不了 LSE 的 backend 只能纯 Ulysses。Ulysses 的 `group_aligned` 策略反而禁止 LSE。
- `backward_op=None` 可以（`_flash_3` 就这么写），但训练路径会缺梯度；推理引擎里通常只需 forward，仍需显式给出 `raise NotImplementedError` 的 backward。

## 6. diffusers 桥

正向（cache-dit → diffusers），`attention/diffusers_bridge.py::_register_cache_dit_attn_backends_to_diffusers`：

- 遍历 `list_bridge_backends()`（即 `bridge_to_diffusers=True` 的那些）。
- 对每个名字调 `_ensure_diffusers_attn_backend(AttentionBackendName, backend)`：diffusers 不认识这个名字时，用 `str.__new__` 手工造一个枚举成员塞进 `_member_map_` / `_value2member_map_`。**这是唯一让自定义名字能被 `model.set_attention_backend("sol_attn")` 接受的机制**，不要删。
- 复制 `constraints` / `_supported_arg_names`（把 `_cp_config` 改名为 `_parallel_config`）/ CP 能力。
- 由 `attention/__init__.py::_maybe_register_custom_attn_backends()` 触发，带 `ENV.CACHE_DIT_CUSTOM_ATTN_ALREADY_DISPATCH` 幂等保护。

反向（diffusers → cache-dit），`register.py::_maybe_register_diffusers_backend_proxy`：

- cache-dit registry 没有该名字、但 diffusers 有时，包一层 `_diffusers_attention_proxy` 复用 diffusers 的实现与约束。
- 这是 `_native_cudnn` 这类"只存在于 diffusers"的后端能跑起来的原因，也是"漏注册被掩盖"的原因。

## 7. 三个对外入口

| 入口 | 代码 | 语义 |
|---|---|---|
| 单卡 / 非并行 | `cache_dit.enable_cache(..., attention_backend="X")` → `caching/cache_interface.py` | 没有 `parallelism_config` 时直接 `set_attn_backend`；`attention_backend is None` 时走 `AttnBackendSelector.auto_select`（目前只处理 NPU） |
| 并行 | `ParallelismConfig(attention_backend="X")` → `distributed/dispatch.py::enable_parallelism` | `_maybe_register_custom_attn_backends()` 后 `set_attn_backend(module, X)`；`ModelMixin` + `None` + `CACHE_DIT`/`NATIVE_HYBRID` 时默认 `"native"` |
| 直接调用 | `cache_dit.set_attn_backend(pipe_or_adapter, "X")` | 见 `interface.py` |

`set_attn_backend` 的两条分支（决定"新 backend 能不能作用于自持模型"）：

- `ModelMixin` 且有 `set_attention_backend` → 调它（diffusers 负责后续 dispatch）。
- 普通 `nn.Module` → 遍历子模块，把 `processor._attention_backend` 与模块自身 `_attention_backend` 都写成名字；一个都没写成就 warning。**非 diffusers 模型只接受 cache-dit enum 里的名字**，否则 `ValueError`。

并行路径还会做一次 CP 校验（`distributed/parallel/diffusers_bridge.py::validate_context_parallel_attention_backend`）：backend 未注册 → 列出兼容集合后 raise；`is_context_parallel_available` 为假 → raise。**这是 "CP 开了但 backend 不支持" 的唯一显式报错点**，所以 `supports_context_parallel` 不能随便标。

## 8. CLI、全局配置与 env

`src/cache_dit/cli/utils.py`：

- `--attn` 的 `choices` 是硬白名单，新增 backend 必须加，否则 argparse 直接拒。
- backend 专属选项统一走 `--attn-params "k=v,k=v"`，合法 key 由 `--attn` 选中的 backend 决定：
  - key 表在 `src/cache_dit/cli/params.py::ATTN_PARAM_SPECS`（按 backend 分表，值是 str→类型的 caster）；
  - `maybe_postprocess_args → _apply_attn_params` 解析后写进进程级配置：`sol_attn` 调 `set_sol_attn_config(**fields)`（不带选项也会调一次，用来提前校验运行时依赖），ffpa 系列对每个 key 调 `set_ffpa_<key>(value)`；
  - key 不属于该 backend、不带 `--attn`、或 backend 没有任何选项 → 启动即报错，不会拖到模型加载后；
  - 新增带选项的 backend：在 `ATTN_PARAM_SPECS` 加一张表，并在 `_apply_attn_params` 里接上它的 setter（命名跟 `set_<backend>_<key>` 对齐就能走 ffpa 同款的循环）。
- 非 sol_attn 的选项会体现在保存文件名后缀里（`strify`，固定顺序 `hybrid_n_early_N` / `hybrid` / `fp4_hadamard` / `fp8_hadamard` / `fp4_pv_mm_type_T` / `fp4_smooth_v` / `fp8_smooth_v`），新增 key 要同步这里。

- 全局配置必须在 **compile / 推理开始前**写入（`set_sol_attn_config` 的 docstring 明确写了 "before inference or compilation"），因为它被后端在每次调用时读全局字典。
- 用户可见的配置函数建议从 `cache_dit.attention` 导出（`set_sol_attn_config` 就在 `__all__` 里）；只给 CLI 用的可以直接 `from ..attention.backends.<name> import set_xxx`（ffpa 系列就是这样）。

env：

| 变量 | 作用 |
|---|---|
| `CACHE_DIT_ATTN_BACKEND` | 进程默认 backend（`_default_active_backend`） |
| `CACHE_DIT_ATTN_CHECKS=1` | 打开 kwargs 过滤 warning 与 `constraints` 执行 |
| `FFPA_CUDA_ALLOW_SMALL_D` | ffpa 私有：放开 D∈[64,256]，否则静默退 SDPA |

## 9. 三个可直接套的模板

### 9.1 最小 backend（无 CP）

```python
from typing import Optional

import torch

from .register import _AttnBackend, _AttnBackendRegistry

try:
  from vendor_attn import vendor_attn_func
except Exception:
  vendor_attn_func = None


@_AttnBackendRegistry.register(
  _AttnBackend.MY_BACKEND,
  constraints=[],
  supports_context_parallel=False,
)
def _my_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  cp_gqa_strategy: Optional[str] = None,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  """Vendor attention backend.

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :param attn_mask: Not supported; pass ``None``.
  :returns: ``[B, N, H, D]``.
  """
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not supported for MY backend.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` is not supported for MY backend.")
  if return_lse:
    raise ValueError("`return_lse` is not supported for MY backend.")
  if vendor_attn_func is None:
    raise RuntimeError("MY backend is not available. Please install `vendor-attn`.")
  out = vendor_attn_func(query, key, value, causal=is_causal, softmax_scale=scale)
  return out.contiguous()
```

### 9.2 多档变体族

一个内部实现 + N 个薄 wrapper（`ffpa.py` 的形状）：

```python
def _impl(query, key, value, attn_mask, is_causal, scale, enable_gqa,
          _cp_config, *, enable_fp8=False, enable_fp4=False, preset=None):
  ...


@_AttnBackendRegistry.register(_AttnBackend.MY_FP8, constraints=[], supports_context_parallel=True)
def _my_fp8_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                      scale=None, enable_gqa=False, return_lse=False,
                      cp_gqa_strategy=None, _cp_config=None):
  return _impl(query, key, value, attn_mask, is_causal, scale, enable_gqa,
               _cp_config, enable_fp8=True)


@_AttnBackendRegistry.register(_AttnBackend.MY_FP8_PER_BLOCK, constraints=[], supports_context_parallel=True)
def _my_fp8_per_block_attention(query, key, value, attn_mask=None, dropout_p=0.0,
                                is_causal=False, scale=None, enable_gqa=False,
                                return_lse=False, cp_gqa_strategy=None, _cp_config=None):
  return _impl(query, key, value, attn_mask, is_causal, scale, enable_gqa,
               _cp_config, enable_fp8=True, preset="per_block")
```

档位配置对象要**缓存**（ffpa 的 `_ffpa_backend_cache`，key 含 device index、全部 override），否则每次调用都重建 dataclass 会进入热路径。

### 9.3 CP backend

```python
def _my_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                  scale=None, enable_gqa=False, return_lse=False,
                  cp_gqa_strategy=None, _cp_config=None):
  if _cp_config is None:
    return _my_attention_op(query, key, value, is_causal)
  return _context_parallel_attention(
    query, key, value,
    None,            # mask 不支持就传 None，并在上面显式 raise
    0.0,
    is_causal, scale, enable_gqa, return_lse, cp_gqa_strategy,
    forward_op=_my_attention_forward_op,
    backward_op=_my_attention_backward_op,
    _cp_config=_cp_config,
  )
```

注意参数是**位置传递**的（`_context_parallel_attention` 的 `forward_op` / `backward_op` / `_cp_config` 是 kw-only），照抄 `sol_attn` / `sage` 的调用形状最稳。
