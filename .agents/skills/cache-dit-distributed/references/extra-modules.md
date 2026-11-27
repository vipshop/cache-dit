# 额外模块：TE-P / VAE-P / ControlNet-P

`transformer` 之外的组件走独立 planner，入口都在 `distributed/dispatch.py::enable_parallelism` 的 `extra_parallel_modules` 循环里。

## 1. 分类与解析

```python
for module in extra_parallel_modules:
  if check_text_encoder(module):   parallelize_text_encoder(module, cfg)     # TP
  elif check_controlnet(module):   parallelize_controlnet(module, cfg)       # CP
  elif check_auto_encoder(module): parallelize_autoencoder(module, cfg)      # DP + tiling
```

- 传模块时按 `_is_text_encoder` / `_is_auto_encoder` / `_is_controlnet`（模块路径判定）分类；传 pipeline 时只看属性名 `text_encoder` / `vae` / `controlnet`。
- 三个 `check_*` 对**已是模块**的入参不会去 pipeline 里找同名属性，所以直接传 `pipe.vae` 是最稳的用法。
- `cache_interface.py` 在调 `enable_parallelism` 前用 `parse_extra_modules(pipe, names)` 把名字解析成模块：名字找不到只 warning 跳过；`text_encoder` 默认会被替换成**参数量最大**的编码器（FLUX 的 `text_encoder_2`），需要原样时用 `skip_text_encoder_override=True`。
- CLI 侧 `cli/utils.py::prepare_extra_parallel_modules` 写的属性名是 `text_encoder` / `vae` / `controlnet`；pipeline 把 VAE 挂在 `video_vae` 这类别名上时只打一条 warning（`parallel-vae is set but no VAE found in the pipeline`）就继续跑，给 pipeline 补 `vae` 别名是标准修法。
- 另外两个断言：`--parallel-params vae=true` 与 `--vae-tiling` / `--vae-slicing` 互斥（VAE-P 自己接管 tile 循环，不能与 diffusers 的 tiling/slicing 叠加）。

## 2. TE-P（`text_encoders/`）

- planner 基类 `TextEncoderTensorParallelismPlanner`：`_apply(text_encoder, config)` 返回 `(module, layer_plans)`，`apply()` 里 `record_plans()` 写 `_rowwise_layers` / `_colwise_layers`。
- 并行度 `parallelism_config.text_encoder_world_size` = `_get_world_size()`（hybrid 下是 `u*r*t`，不是 `tp_size`）。所以「4 卡 ulysses_tp」的 TE 会用 4 卡 TP，而 transformer 只有 2 卡 TP。
- 虽然名字叫 TE-P，实现就是 TP（注释明确："Currently, only support tensor parallelism or data parallelism for extra modules"），mesh 是独立 1D `init_device_mesh([text_encoder_world_size])`，与 transformer 的 tp mesh 无关。
- 已注册（按类名前缀）：`T5EncoderModel`、`UMT5EncoderModel`、`Qwen2_5_VL*`、`Qwen3Model` / `Qwen3ForCausalLM` / `Qwen3VL*`、`Qwen3_5Model`、`Llama*`、`Mistral*`、`Gemma*` / `T5Gemma*`、`Glm*` / `Glm4*`、`SmolLM3*`、`MingImageMLLM`、`GlmImageForConditionalGeneration`。
- 大编码器（几十 GB）仍要**先全量加载再切**，峰值是「完整权重 + 已切好的 transformer」，这是很多 OOM 的根因；解锁方向是 meta-device 分片加载，目前没有。

## 3. VAE-P（`autoencoders/`）

### 3.1 结构

- planner 基类 `AutoEncoderDataParallelismPlanner`：`mesh()` 用 `auto_encoder_world_size`（同样是 `_get_world_size()`）建 1D DP mesh。
- 思路是 **tile 级 DP**：`auto_encoder.enable_tiling()` 后，按 `count % world_size == rank` 把 tile 轮转分给各 rank 编码/解码，然后 **rank 0 收齐所有 tile、做 `blend_v`/`blend_h` 拼接**，再把结果沿 rank 链 `rank → rank+1` 接力传回所有 rank。
- 每个模型几何不同（`tile_sample_min_size` / `tile_latent_min_size` / `tile_overlap_factor`），所以是一模型一个 planner：`AutoencoderKL`、`AutoencoderKLWan`、`AutoencoderKLHunyuanVideo`、`AutoencoderKLFlux2`、`AutoencoderKLLTX2Video`、`AutoencoderKLQwenImage`（含 21）、`AutoencoderKLMiniMaxH3`（含 image）、`AutoencoderKLPH3Image`。
- 输出必须**逐比特**与单卡一致（tile 几何与 blend 顺序都不能动）；`test_*_vae_dp.py` 就是干这个的。

### 3.2 tile P2P 协议（`autoencoders/utils.py`）

两套实现，**不能混用**：

| 类 | 行为 | 用在哪 |
|---|---|---|
| `TileBatchedP2PComm` | shape 立即发，payload 进队列等 `commit()`/`sync()` 批量发 | 单 tile 相位，吞吐高 |
| `TileSafeP2PComm` | 每条消息 `send(shape) → send(payload)` 完整走完再发下一条；`commit/wait/sync` 是 no-op | **多 tile decode 必须用它** |

设计原因（写在类的 docstring 里，都是实测结论）：

- **`nccl` 会忽略 `tag`**：`isend(tag=1)` 之后 `isend(tag=0)`，接收端按发送顺序收，所以「shape 走一个 tag、payload 走另一个 tag」不成立。
- 旧实现 shape 立即发、payload 延迟到 `commit()`，两个 `recv_tensor` 连调时第二个 shape recv 会吃掉第一个 payload，把 float32 bit pattern 当 shape → `SymIntArrayRef expected to contain only concrete integers`（gloo 上是 "Received data size doesn't match expected size"）。
- **dtype 必须上线**：shape 的 int64 消息尾部追加一个 dtype code（`_DTYPE_CODES`），接收端以对端宣布的 dtype 为准；否则 bf16 tile 被 float32 buffer 接走会静默错或挂到 NCCL 看门狗。
- `set_shape` / `clear_shape` 在 `TileSafeP2PComm` 里是**弃用的 no-op**：缓存 shape 会在两个 rank 不同步时把协议永久带偏，40 字节不值得。
- dims 可以只发一次（`_s_dims`），因为各 rank 的维度数相同且 planner 只设一次；shape 每次都要发。

历史：顺序问题由 `2800a6dc fix(vae): add an order-safe tile transport for multi-tile VAE decode` 修复，回归用例 `tests/parallelism/test_vae_p2p_order.py`。

### 3.3 常见失败

| 症状 | 原因 |
|---|---|
| `--parallel-params vae=true` 无效果 | pipeline 没有 `pipe.vae` 属性（见 §1） |
| decode 阶段挂死 / 600s NCCL 超时 | 两 rank 消息数不一致（某 rank 提前 return）、或 dtype 不一致 |
| `SymIntArrayRef ... concrete integers` | 用了 `TileBatchedP2PComm` 跑多 tile decode |
| 图像出现十字接缝 | tile 几何被改（overlap 太小/为 0）；不要为了速度去掉 overlap |
| 只有 rank 0 有图 | 接力链 `rank → rank+1` 断了（`comm.sync()` 前后顺序错） |

## 4. ControlNet-P（`controlnets/`）

- 只支持 CP，且要求 `backend == CACHE_DIT`；其它后端进来只 warning 并跳过。
- `parallelize_controlnet` 与 transformer 同样的 `_ContextParallelConfig` + `ControlNetContextParallelismPlannerRegister`；`cp_plan` 也可用户自定义。
- pipeline 有 controlnet 时 `cache_interface.py` 会置 `parallelism_config._has_controlnet = True`（用户可显式覆盖），部分模型（Z-Image-ControlNet）据此走专门的 plan。
- 已注册：`zimage_controlnet.py`（`ZImageControlNet*`）。

## 5. 清理

`remove_parallelism_stats(transformer)` 只递归**一层** extra modules（注释明确"extra parallel modules should not have nested extra parallel modules"），并删除每个模块的 `_is_parallelized` / `_parallelism_config` / `_extra_parallel_modules`。TE/VAE 的 `_is_parallelized` 由各自 `parallelize_*` 设置，所以清理要从 transformer 入口进。
