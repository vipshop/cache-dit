# Backend 契约与现有能力矩阵

## 1. 唯一合法的函数签名

```python
def _<name>_attention(
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
) -> torch.Tensor: ...
```

- 11 个参数**全部声明**，即使不支持也要在函数体里 raise（`sage` / `sage3` / `sol_attn` 都是这个形状）。少声明 = dispatcher 静默丢弃语义。
- 返回：`torch.Tensor`，或 `return_lse=True` 时 `(out, lse)`。
- 返回 layout 必须与输入一致（NHD `[B, N, H, D]`）。形状检查只在 `CACHE_DIT_ATTN_CHECKS=1` 时由 `_check_shape` / `_check_device` / `_check_qk_dtype_bf16_or_fp16` 执行。
- `query` 与 `key`/`value` 的 `N` 可以不同（cross-attention），但 `H` 在第 -2 维；`enable_gqa` 时 `H_q != H_kv` 也是合法的（`_check_shape` 只校验 D 与 Nkv 一致）。

## 2. Layout

cache-dit 的公共约定是 **diffusers 的 NHD `[B, N, H, D]`**，不是 FlashAttention 的 BHND `[B, H, N, D]`。转换责任在 backend 内部：

| 做法 | 先例 | 代价 |
|---|---|---|
| vendor 直接吃 NHD | `sage`（`tensor_layout="NHD"`）、`flash`、`flash_varlen` | 零拷贝 |
| 调 vendor 前 `permute(0,2,1,3)`，回来再 permute | `sage3`（`transpose+contiguous`）、`native`（`permute` 视图）、`cudnn` | 需要 contiguous 时是一次真实拷贝 |
| 按卡型选快路径 | `ffpa`：sm_120 且无 mask 时走 NHD 零拷贝，否则退回 BHND + `contiguous()` | 两条路径都要有数值等价的测试 |

`ffpa` 的 NHD 快路径还有一个附加约束：上游 persist-D kernel 要求 `Nq`/`Nkv` 足够大（`8 <= Nq < 512 or Nkv < 512` 会被拒），所以它捕获 `TypeError` 中的 `"requires the inference fast path"` 再退 BHND。**退路必须打 `warning_once` 并带 shape**，否则短序列会悄悄慢。

## 3. dtype

- 绝大多数 kernel 只吃 fp16 / bf16。`--attn-params type=X` 是 pipeline 级设置，会落到跑 fp32 的组件（例如视频 VAE decoder），所以**必须处理非 fp16/bf16 输入**：
  - 首选：退 SDPA（`ffpa`），但**有 CP 时必须 raise**，因为 SDPA 只算本地分片，结果静默错。
  - 次选：显式 raise，让用户自己换 backend。
- bf16 输入 + fp32 权重（autocast 关掉时）会导致 kernel 报错；不要靠 `torch.autocast` 兜底。

## 4. `attn_mask`

SDPA 语义：`bool` 时 `True` 表示参与，`float` 表示加性 bias。cache-dit 不解释语义，只看形状：

- `ffpa` 把 `[Nkv]` / `[B, Nkv]` 扩成 4D，3D/4D 原样透传；末维必须等于 `key` 的序列长度；`attn_mask` 与 `is_causal=True` 互斥。
- `flash_varlen` 把 2D/3D mask 折成 2D 后 unpad（`_mask_to_2d` + `_get_unpad_data`），只支持"尾部 padding"这一类 mask。
- `native` / `cudnn` / `npu` 直接把 mask 喂给 SDPA 系实现。
- 其余 backend 一律 raise。

**CP 与 mask 是互斥的**，除非把 `forward_op.__name__` 加进 `_context_parallel_attention` 的 allowlist。当前 allowlist 只有 `_native_attention_forward_op`、`_cudnn_attention_forward_op`、`_flash_varlen_attention_forward_op`、`_npu_attention_forward_op`、`_npu_fused_infer_attention_forward_op`。即使加进去，分片语义（哪个 shard 被 mask、mask 在 all-to-all 之后对不对）仍要自己证明。

## 5. `return_lse` 与 CP 拓扑

| CP 拓扑 | 需要 LSE | 说明 |
|---|---|---|
| Ring（`ring_degree>1`） | **必须** | `RingAttention.forward` 固定传 `return_lse=True`，用 LSE 做 log-sum-exp 合并 |
| Ulysses（`ulysses_degree>1`，纯） | 不需要 | 序列切分后各 rank 结果直接 concat |
| USP（Ring+Ulysses） | **必须** | 内含 Ring |
| `cp_gqa_strategy="group_aligned_flash_varlen"` | 不允许 | `ulysses.py` 显式 raise "return_lse is not supported for group-aligned GQA Ulysses" |

所以：

- 不能返回 LSE 的 backend（`sol_attn`、`sage3`、`flash_varlen`、cudnn 路径）只能声明纯 Ulysses。`sol_attn` 甚至在 `ring_degree>1` 时直接 `NotImplementedError`，因为本地 KV 分片会改变它的路由统计量。
- LSE 形状：`[B, N, H]`；Ring 会自己 `unsqueeze(-1)` 成 `[B, N, H, 1]`。
- `lse` 必须是**自然对数底**（与 `F.sigmoid(lse - prev_lse)` 的合并公式一致）。若 vendor 用 log2，必须换算。

## 6. GQA

| 做法 | 先例 |
|---|---|
| 原生 `enable_gqa=True` 透传给 SDPA | `native` / `cudnn` / `ffpa` |
| vendor 自己按 shape 判断，不用 `enable_gqa` | `sage`（注释写得很明确） |
| 显式拒绝 | `flash`（2/3）、`sage3`、`sol_attn` |
| 靠 `cp_gqa_strategy` 在 CP 下对齐 head group | `flash_varlen`（`replicate_kv_sequence` / `group_aligned_flash_varlen`） |

注意 `enable_gqa` 只在 torch>=2.5 时被 dispatcher 透传，所以签名默认值必须是"安全语义"，不能把 GQA 假设写死在默认路径里。

## 7. 现有 backend 能力矩阵

| backend | 来源 | mask | CP | LSE | GQA | 硬件/精度门 |
|---|---|---|---|---|---|---|
| `native` | SDPA | ✅ | ✅ 三种 | ❌ | ✅ | 通用 |
| `_sdpa_cudnn` | SDPA + CUDNN kernel | ✅ | ✅ | ❌ | ✅ | 需 cuDNN SDPA |
| `_native_cudnn` | 仅 diffusers（proxy） | ✅ | 跟随 diffusers | — | ✅ | 不在 cache-dit enum |
| `flash` | flash-attn 2 | ✖️ raise | ✅ | ✅ | ✖️ raise | fp16/bf16 |
| `flash_varlen` | flash-attn varlen | ✅ 2D/3D | ✅ | ✖️ | 策略式 | fp16/bf16 |
| `_flash_3` | FA3 | ✖️ raise | ✅ | ✅ | ✖️ raise | Hopper |
| `sage` | SageAttention-2 | ✖️ raise | ✅ | ✅ | vendor 自判 | fp16/bf16 |
| `sage3` | SageAttention-3 | ✖️ raise | ✅ | ✖️ | ✖️ raise | Blackwell；head_dim 64/128 |
| `sol_attn` | NVlabs Sol-Attn | ✖️ | 仅 Ulysses | ✖️ | ✖️ raise | BF16、head_dim=128、等长、默认 scale |
| `ffpa` | ffpa-attn fp16/bf16 | ✅（单卡） | ✅ | ✖️ | ✅ | sm_120 / sm_89 |
| `ffpa_fp8` | ffpa-attn FP8 | ✅（单卡） | ✅ | ✖️ | ✅ | sm_120 / sm_89；D=192 有已知上游错误 |
| `ffpa_fp8_per_block` | ffpa-attn FP8 per-block | ✅（单卡） | ✅ | ✖️ | ✅ | 同上；最快、精度最低 |
| `ffpa_fp4` | ffpa-attn NVFP4 | ✅（单卡） | ✅ | ✖️ | ✅ | 仅 sm_120 |
| `_native_npu` / `_npu_fia` / `_mindiesd_laser` | Ascend NPU | ✅ | ✅ | ✖️ | 视实现 | NPU |

填这张表就是填 `docs/user_guide/ATTENTION.md` 的两列（parallelism / attn_mask），**不能写"支持"却没测过**——用户是按这张表选 backend 的。

## 8. 近似 backend 的额外契约

`sol_attn` 这类不逐比特的实现，除了函数签名还要有：

1. **eligibility 判定**（`_ineligible_reason`）：把每一条"不支持"写成字符串 reason，顺序固定、可测。
2. **fallback 策略**：默认退 `_native_attention` 并 `_record_fallback(reason)`（计数 + 首次 warning），`strict=True` 时改成 raise。**fallback 必须可观测**，否则等于静默降级。
3. **stats 接口**：`kernel_calls` / `fallback_calls` / `contiguous_copies` / `fallback_reasons`，供验收断言"真的走了 kernel"。
4. **不做比做更好时的说明**：把精度限制写进 docstring 的 warning block（`sage3` 用一段 ASCII 数据流解释 FP4 P 只有 `{0,0.5,1}` 三档、cos≈0.03），并在 `ATTENTION.md` 里给出推荐替代（`--attn-params type=sage`）。
