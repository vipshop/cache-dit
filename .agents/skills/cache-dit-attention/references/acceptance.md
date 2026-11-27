# 验收：测试骨架、假绿清单、性能与精度口径

运行环境固定 `conda activate cdit`，cache-dit 默认用 4/5 号卡：

```bash
conda run -n cdit python -m pytest tests/api/test_<name>_backend.py -q
conda run -n cdit python -m pytest tests/parallelism/test_<name>_cp.py -q
```

## 1. 必写测试（照现有文件的形状抄）

| 文件 | 覆盖 | 先例 |
|---|---|---|
| `tests/api/test_<name>_backend.py` | 注册 / 签名 / 桥接 / 缺依赖 / 语义拒绝 / 数值 / 编译 | `tests/api/test_sol_attn_backend.py` |
| `tests/parallelism/test_<name>_cp.py` | 2/4 卡 Ulysses（+ Ring 若声明）与单卡对比 | `tests/parallelism/test_sol_attn_cp.py` |
| `tests/parallelism/test_attention_dispatch.py` | 只在动了 bridge / dispatcher 语义时追加 | 同名文件 |
| `tests/api/test_set_attn_backend.py` | 只在改了 `set_attn_backend` 分支时追加 | 同名文件 |

### 1.1 注册与签名

```python
def test_my_backend_registration_and_full_signature():
  assert _AttnBackend.MY_BACKEND.value == "my_backend"
  assert _AttnBackendRegistry.get_backend("my_backend") is not None
  assert _AttnBackendRegistry.is_context_parallel_available("my_backend") is EXPECTED_CP
  required = {
    "attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa",
    "return_lse", "cp_gqa_strategy", "_cp_config",
  }
  assert required <= _AttnBackendRegistry.get_supported_arg_names(_AttnBackend.MY_BACKEND)
```

这条同时守住"漏声明参数"这个最常见的静默 bug。`EXPECTED_CP` 必须与 `supports_context_parallel` 的真实能力一致，不许两头写不同。

### 1.2 diffusers 桥

```python
def test_my_backend_diffusers_registration():
  from cache_dit.attention import _maybe_register_custom_attn_backends
  _maybe_register_custom_attn_backends()
  from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
  assert _AttentionBackendRegistry._backends.get(AttentionBackendName("my_backend")) is not None
```

如果名字是 diffusers 已有的（`flash` / `sage`），再加一条"`set_attn_backend(dummy ModelMixin)` 收到该名字"。

### 1.3 依赖缺失必须是明确错误

```python
def test_my_backend_missing_dependency_is_clear(monkeypatch):
  monkeypatch.setattr(my_backend, "vendor_attn_func", None)
  with pytest.raises(RuntimeError, match="my_backend.*not installed"):
    _dispatch_attention_fn(q, k, v, backend="my_backend")
```

### 1.4 语义拒绝逐条测

按签名逐个参数构造非法调用：`attn_mask` / `is_causal=True` / `dropout_p=0.1` / `enable_gqa=True` / `return_lse=True` / 非默认 `scale` / 错误 dtype / 错误 head_dim。每条都要 `pytest.raises(..., match="<原因片段>")`，并且**在 `strict` 语义存在时同时测 strict 与 fallback 两种模式**。

### 1.5 数值

两级：

1. 与独立 reference 比（`sol_attn` 的 `_sol_reference` 手写 online-softmax，是最高强度的做法），或与 dense SDPA 比。
2. 断言用 `relative L2 <= 3e-3` + `cosine >= 0.999`（`sol_attn` 的口径；1024 token 以上放宽到 4e-3）。

低精度 backend 到不了这个门时必须**显式降低门并写理由**（`sage3` 的 FP4 P 只有三档，cos≈0.03），不许把门改到能过为止而不解释。

"全 exact 档应接近 dense"是一条很有价值的写法（`test_sol_attn_all_exact_is_close_to_dense`：把 `tau` 推到 `-1e9` 让所有块都走 exact），它把"路由逻辑"和"数值实现"解耦验证。

### 1.6 编译

包了 `torch.library` 就必测：

```python
  torch.library.opcheck(torch.ops.<ns>.<op>, tuple(args))
  compiled = torch.compile(lambda a, b, c: _dispatch_attention_fn(a, b, c, backend="my_backend"),
                           fullgraph=True)
  torch.testing.assert_close(compiled(q, k, v), eager, rtol=0, atol=0)
```

`fullgraph=True` + 逐比特断言才算过。没有 fake impl 的算子会在 tracing 阶段直接失败。

### 1.7 CP

```python
def _run_ulysses(rank, world_size, port):
  ...
  cp_config = _ContextParallelConfig(ring_degree=1, ulysses_degree=world_size)
  cp_config._ulysses_mesh = mesh["ulysses"]
  full = _dispatch_attention_fn(q, k, v, backend="my_backend")
  shard = _dispatch_attention_fn(q[:, local], k[:, local], v[:, local],
                                 backend="my_backend", cp_config=cp_config)
  assert _relative_l2(shard, full[:, local]) <= 3e-3
```

- 端口用 `socket.bind(("127.0.0.1", 0))` 取空闲端口，并 `try/finally: destroy_process_group()`；写死端口会让整文件串行跑时随机 `EADDRINUSE`。
- 声明 Ring 支持的再加一条 Ring 用例；只支持 Ulysses 的加一条"Ring 必须 `NotImplementedError`"。
- 想保证"分片语义对"而不只是数值接近，用 spy 记录 backend 实际看到的 mask / 分片，断言 `masks[rank] == full_mask[:, rank:rank+1]`（`test_sol_attn_cp.py` 的做法）。

## 2. 假绿清单（这些都是"测试全绿但 backend 没生效"）

1. **走了 fallback**：数值与 SDPA 完全一致（rel-L2 恰好 0）反而是可疑信号。必须断言 `kernel_calls > 0 and fallback_calls == 0`，或 spy vendor 调用次数。
2. **`--attn-params type=X` 没落到模块上**：非 diffusers 模型若没有 `_attention_backend` 属性，`set_attn_backend` 只 warning。验收时读回 `processor._attention_backend`。
3. **CP 路径没被走到**：`cp_config=None` 时永远走单卡分支。CP 测试必须真的传 `_ContextParallelConfig` 并 `dist.init_process_group`。
4. **mask 被丢弃**：签名漏 `attn_mask` 时 dispatcher 静默过滤，输出看起来"没报错"。用 `attn_mask` 全 `-inf` / 全 False 的极端 mask 断言输出变化。
5. **compile 没编译**：`TORCH_LOGS=graph_breaks` 打 0 条不代表没跑 dynamo；判据是 `torch._dynamo.utils.counters["stats"]["unique_graphs"]`。
6. **PSNR 量纲错**：`cache_dit.metrics.calculate_psnr` 的 peak 固定 255，喂 float[0,1] 会白送 +48.13 dB。自己按 `data_range=1` 算，或先转 uint8。

## 3. 性能口径

- 计时用 CUDA event，不用 `perf_counter` 量 launch（会把 launch 时间算进去）。
- 同 recipe、同 seed、同分辨率，比较 **attention 段耗时**与端到端；attention 占比决定收益上限：
  - 1024² 图像模型 attn 通常只占 18–25% ⇒ 端到端 1.03–1.05x 是正常结果；
  - 长视频 / 长序列（H3 类 56%+）才有明显收益。
- 与 `--compile`、`--cache-params`、量化叠加时，各行独立报，不要用一个加速比概括（cache 会减少前向次数，直接放大或稀释 attention 的绝对收益）。
- 报告必须写清实际生效档位（backend 名 + 精度档 + 硬件 capability），因为同一 backend 名在不同卡上走的是不同 kernel 家族。

## 4. 精度口径

- 参照物固定为"同模型、同 seed、同 recipe 的 native SDPA"，不给跨模型阈值。
- 端到端差异要和**同配置重跑的噪声地板**比，不能和 0 比；`--compile` 路径 run-to-run 不收敛，只能报区间。
- 指标至少给 PSNR + SSIM，二者会背离；再补一次视觉检查（局部花屏在 PSNR 上常常看不出来）。
- 低精度档（fp8/fp4）必须给"退化来源"的定位：是 QK 量化、PV 量化、还是 P 的量化（`sage3` 的 cos≈0.03 就是 P 的 E2M1 造成的）。

## 5. 交付表

报告用这两张表，缺列视为没做完：

| 项 | 证据（命令 / 文件 / 断言名） | 结论 |
|---|---|---|
| 注册 + 签名 | `test_<name>_backend.py::test_<name>_registration_and_full_signature` | |
| diffusers 桥 | `...::test_<name>_diffusers_registration` | |
| 真实调用 | `kernel_calls` / spy 计数 | |
| CP | `test_<name>_cp.py`（world_size=2/4） | |
| 编译 | `opcheck` + `fullgraph` 逐比特 | |
| e2e 精度 | native vs 新 backend PSNR/SSIM | |

| 配置（backend / 精度档 / 卡型） | attention 段 | 端到端 | 相对 native |
|---|---|---|---|
| | | | |

## 6. pre-commit

提交前按仓库既有要求跑 pre-commit 并修完；不要用 `--no-verify` 绕过（那是 LeetCUDA 仓库的约定，不适用于 cache-dit）。
