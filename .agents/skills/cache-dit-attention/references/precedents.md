# 先例拆解：ffpa-attn / SageAttention-2 / Sol-Attn / 其余

每个先例按 "它解决什么问题 → 抄什么 → 别抄什么" 写。路径相对 `src/cache_dit/attention/backends/`。

## 1. ffpa-attn：多档变体族（最完整的工业形态）

文件 `ffpa.py`，四个 backend：`ffpa` / `ffpa_fp8` / `ffpa_fp8_per_block` / `ffpa_fp4`。

### 1.1 结构

```text
_ffpa_attention              → _ffpa_attention_impl(enable_fp8=False, enable_fp4=False)
_ffpa_fp8_attention          → _ffpa_attention_impl(enable_fp8=True)
_ffpa_fp8_per_block_attention→ _ffpa_attention_impl(enable_fp8=True, fp8_preset="per_block")
_ffpa_fp4_attention          → _ffpa_attention_impl(enable_fp4=True)
```

`_ffpa_attention_impl` 做四件事：语义门（dropout / lse / mask+causal 互斥）→ dtype 分流（非 fp16/bf16 退 SDPA，有 CP 则 raise）→ 硬件门（`_require_supported_cuda`）→ 组 CP 或直调 `_ffpa_attn_core`。

### 1.2 值得抄的点

- **能力门按精度族分开写**：fp16/fp8 允许 `sm_120` 与 `sm_89`，fp4 只允许 `sm_120`。错误信息里直接写"sm_90 / sm_100 are not supported"，不要让用户在 kernel 崩溃里猜。
- **配置对象缓存**：`_build_ffpa_cuda_backend` 用 `(device.index, flags..., overrides...)` 作 key 缓存 `CUDABackend`（`__post_init__` 有 auto-resolve 开销）。多档变体族必备。
- **全局 override + CLI 交叉校验**：`set_ffpa_hybrid` / `set_ffpa_fp4_hadamard` / `set_ffpa_fp4_pv_mm_type` 等写模块级全局变量，CLI 侧逐个校验"必须配对应 `--attn-params type=X`"，再在 `_build_ffpa_cuda_backend` 里注入 kwargs。**新字段用 `kwargs["x"] = True` 后置注入**，这样旧版 ffpa-attn 没有该字段时 off-path 不会炸。
- **mask 归一化集中在 backend**：`_normalize_attn_mask` 只扩展无歧义的低秩形状（`[Nkv]` → `[1,1,1,Nkv]`、`[B,Nkv]` → `[B,1,1,Nkv]`），3D/4D 原样透传，并在 docstring 里写"mask 语义是调用方的契约"。
- **NHD 零拷贝快路径**：sm_120 且无 mask 时直接 `tensor_layout="NHD"`，省掉输入 permute 视图和输出 permute（非连续视图会在下游 flatten 时触发真实拷贝）。退路是同一数学的 BHND permute。

### 1.3 别抄的点

- **共享对象上的可变状态**：`backend.tensor_layout = "NHD"` 改的是缓存里的同一个 `CUDABackend`。它靠"每个 fallback 分支都显式写回 `"HND"`"来维持正确性——这是能工作的，但属于脆弱约定，新 backend 若没有等价必要性，把 layout 放进调用参数而不是对象状态。
- **依赖上游静默回退**：ffpa-attn 在 D<64 时会悄悄退 SDPA，所以文件顶部必须 `os.environ.setdefault("FFPA_CUDA_ALLOW_SMALL_D", "1")`。如果你接入的 vendor 也有这种"条件不满足就换实现"的行为，把它变成显式分支 + 日志。
- **把能力门写在调用体而不是 `constraints`**：这不是缺点，是现实（`constraints` 默认不执行）。别以为 `constraints=[]` 就代表"没有约束"。

## 2. SageAttention-2：最小可用样板

文件 `sage.py`，全文不到 130 行。

```python
@_AttnBackendRegistry.register(_AttnBackend.SAGE, constraints=[], supports_context_parallel=True)
def _sage_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                    scale=None, enable_gqa=False, return_lse=False,
                    cp_gqa_strategy=None, _cp_config=None):
  ...
  out = sageattn(q=query, k=key, v=value, tensor_layout="NHD",
                 is_causal=is_causal, sm_scale=scale, return_lse=return_lse)
```

抄什么：

- vendor 支持 NHD 就直接传 NHD，不做 permute。
- GQA 交给 vendor 自己判断（`sageattn` 会看 Q/K 的 head 数），不传 `enable_gqa`，但注释里写清原因。
- CP 走 `_context_parallel_attention(forward_op=_sage_attention_forward_op, ...)`，forward op 里返回 `lse.permute(0, 2, 1)` —— **vendor 的 LSE 常常是 `[B, H, N]`，契约要的是 `[B, N, H]`**。
- 依赖缺失在调用时报 `RuntimeError("... Please install `sageattention` to use it.")`，import 期只 `sageattn = None`。

别抄：

- 顶层和 forward op 里各写一遍同样的检查（`attn_mask` / `dropout_p` 重复 raise）。可以保留，但要意识到 forward op 的检查是 CP 路径唯一生效的那一份——顶层检查在 CP 分支之后就被绕过了。
- `return_lse` 在顶层和 forward op 里的处理分叉（顶层 `out, lse, *_ = out`，forward op 里 `lse.permute(0,2,1)`），新 backend 建议统一成一个内部函数。

## 3. Sol-Attn：近似稀疏 backend 的完整样板

文件 `sol_attn.py`，约 370 行，是本仓库唯一把"近似 + 可观测 fallback"做全的先例。

### 3.1 三层结构

| 层 | 内容 |
|---|---|
| 运行期校验 | `_validate_sol_runtime`（包存在、`torch>=2.10`、`cuda>=12.8`、`triton>=3.6`、cute backend 还要 `nvidia-cutlass-dsl>=4.5` + `cuda-python`）与 `_validate_sol_device`（capability>=8.0，按 device index 缓存上游 backend 名） |
| 自定义算子 | `torch.library.define("cache_dit_sol_attn_ops::forward", "(Tensor q, Tensor k, Tensor v, float scale, float tau, int thresh_type, int sink_tokens) -> Tensor")` + `impl("CUDA")` + `register_fake` |
| 合法性判定 | `_ineligible_reason(...)` 返回字符串 reason；不合法时 `_record_fallback(reason)` 后调 `_native_attention`；`strict=True` 改成 raise |

### 3.2 为什么必须包 `torch.library`

上游是 Triton + CuTe DSL kernel。直接调会被 dynamo trace 进 vendor 代码，inductor 重编译会遇到上游不满足的约束（`tl.arange` 必须 2 的幂等），表现为碎图或编译失败。包成自定义 op 后 inductor 视其为黑盒，**并且 fake impl 让它能在 meta device 上跑通 tracing**：

```python
@torch.library.register_fake(_SOL_ATTN_OP_NAME)
def _sol_attn_op_fake(q, k, v, scale, tau, thresh_type, sink_tokens):
  return torch.empty(q.shape, dtype=q.dtype, device=q.device)
```

配套测试（`tests/api/test_sol_attn_backend.py`）：

```python
  torch.library.opcheck(torch.ops.cache_dit_sol_attn_ops.forward, (q, k, v, scale, 1.0, 0, 128))
  compiled = torch.compile(lambda a, b, c: _dispatch_attention_fn(a, b, c, backend="sol_attn"),
                           fullgraph=True)
  torch.testing.assert_close(compiled(q, k, v), eager, rtol=0, atol=0)
```

### 3.3 eligibility 要写成可测的字符串

```python
def _ineligible_reason(query, key, value, attn_mask, dropout_p, is_causal,
                       scale, enable_gqa, return_lse, cp_gqa_strategy):
  if any(isinstance(x, DTensor) for x in (query, key, value)):
    return "DTensor inputs require conversion to local torch.Tensor first"
  ...
  if query.shape[-1] != 128:
    return f"Sol-Attn requires head_dim=128, got {query.shape[-1]}"
```

抄这个形状：每条 reason 是一个可断言的前缀，测试里 `with pytest.raises(ValueError, match="default scale")` 直接命中。

### 3.4 fallback 与 strict

```python
  if reason is not None:
    if _SOL_CONFIG["strict"]:
      raise ValueError(f"Sol-Attn is ineligible: {reason}.")
    _record_fallback(reason)
    return _native_attention(query=query, key=key, value=value, ...)
```

`_record_fallback` 做三件事：`fallback_calls += 1`、按 reason 计数、每个 reason 只 warning 一次。`_get_sol_attn_stats()` 把 `kernel_calls` / `fallback_calls` / `contiguous_copies` / `fallback_reasons` 暴露给验收——**这是"backend 真的被用上"的直接证据**。

### 3.5 配置与 CP

- `set_sol_attn_config` 在 `cache_dit.attention.__init__` 导出，docstring 明确"必须在推理或编译之前调用"。
- 只允许纯 Ulysses：`ring_degree > 1` 直接 `NotImplementedError`，因为本地 KV 分片会改变路由统计量与近似分母。
- CP 分支的 `forward_op` 里 `return_lse` 直接 raise，CP 调用时 `return_lse=False` 固定传入。

### 3.6 别抄

- 不合法时退 native 是"默认可用"的取舍，不是通用推荐：如果你的 backend 与 dense 结果**形状上就不兼容**（比如改变了序列长度），fallback 会掩盖真实 bug，应该直接 raise。
- `scale` 只接受默认值（`_ineligible_reason` 里用 `math.isclose(rel_tol=1e-6)`）。这是它当前实现的限制，不是通用做法；能支持任意 scale 就别拒绝。

## 4. SageAttention-3：形状适配与编译黑盒

文件 `sage3.py`。

- vendor 要求 BHND，cache-dit 是 NHD → op 内 `transpose(1, 2).contiguous()`，输出再转回。
- 只支持 `head_dim ∈ {64, 128}` → 用零 pad 到最近的合法尺寸（`D<=64 → 64`，`D<=128 → 128`），`D>128` 退 SDPA。数学上 QK 点积不变，softmax scale 从 `1/sqrt(D)` 变成 `1/sqrt(D_pad)`，V 的 pad 维为 0 所以输出切片是精确的。
- 同样用 `torch.library` 包黑盒（理由同 Sol-Attn 的 Triton 问题）。
- **精度警告写在 docstring 里**：FP4 的 P 矩阵在 `[0,1]` 只有 `{0, 0.5, 1}` 三档，PV MMA 实际是在三桶权重上算，与 SDPA 的 cos≈0.03；README 建议交替使用 Sage2++，所以 cache-dit 推荐 `--attn-params type=sage` 而不是 `--attn-params type=sage3`。

别抄：`_sage3_attention` 的 `scale` 参数**没有往下传**（静默忽略非默认 scale）。新 backend 要么透传、要么 raise。

## 5. flash_varlen：mask → unpad 与 GQA 策略

文件 `flash_varlen.py`。

- mask 支持是"折成 2D padding mask 再 unpad"：`_mask_to_2d`（3D 时取对角线求和推长度）+ `_get_unpad_data`（`seqlens_in_batch` / `cu_seqlens` / `max_seqlen`）。只对尾部 padding 有效。
- 因为 mask 是它的核心能力，`_flash_varlen_attention_forward_op` 被加进了 `_context_parallel_attention` 的 mask allowlist。
- GQA 在 CP 下分两种策略：`replicate_kv_sequence`（把 KV 复制成 MHA 布局）、`group_aligned_flash_varlen`（head group 对齐的 varlen），由 `cp_gqa_strategy` 传入；两者都不支持 `return_lse`。

## 6. `_sdpa_cudnn` / `_native_cudnn`：两种名字，两条注册路径

- `_sdpa_cudnn` 在 cache-dit enum 里，实现是 `with torch.nn.attention.sdpa_kernel(SDPBackend.CUDNN_ATTENTION): F.scaled_dot_product_attention(...)`，支持 mask，CP 用 `_cudnn_attention_forward_op`（也在 mask allowlist 里）。
- `_native_cudnn` **只在 diffusers 里**。CLI 能列它，是因为 `set_attn_backend` 对 diffusers 模型直接把字符串透传给 `set_attention_backend`；非 diffusers 模块会 `ValueError("Non-diffusers modules only support cache-dit attention backends")`。若你要接的"backend"只是 SDPA 的一个 kernel 选择，而且只想给 diffusers 模型用，就走这条 proxy 路线，不要污染 enum。

## 7. NPU：平台原生 backend

`npu.py` 里 `_native_npu` / `_npu_fia` / `_mindiesd_laser` 都是 `supports_context_parallel=True` + mask 支持，靠 `torch_npu` 提供实现。平台自动选择在 `backend_selector.py::AttnBackendSelector.auto_select`：检测到 `device.type == "npu"` 时返回 `_native_npu`。

要接 AMD / 其他加速器时照这个模式：**先有能跑的 backend，再谈 auto_select**；auto_select 目前是 global class state（`_selected` 置位后不再重算），多设备混合场景不要指望它。
