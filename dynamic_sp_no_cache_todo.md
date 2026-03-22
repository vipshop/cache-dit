# Dynamic SP No Cache - TODO 开发计划

> 参考 `dynamic_sp_no_cache_94970bd2.plan.md` 拆解的可执行开发清单。  
> 目标：实现 per-step 动态 SP degree（Ulysses/Ring/USP），仅在 parallelism 层实现，不接入 cache。

## 0) 预检查与范围确认（P0）

- [ ] 确认当前已有改动文件：`src/cache_dit/parallelism/__init__.py`、`src/cache_dit/parallelism/config.py`
- [ ] 确认本次实现范围仅为 parallelism 层（无 cache 逻辑改动）
- [ ] 确认运行环境可用 `torch.distributed` 与 `DeviceMesh`

**验收标准**
- 变更范围限定在计划定义文件内，不引入额外模块耦合。

---

## 1) 新建核心模块 `dynamic_sp.py`（P0）

- [ ] 新建 `src/cache_dit/parallelism/dynamic_sp.py`
- [ ] 实现 `DynamicSPConfig`:
  - [ ] `enabled: bool = False`
  - [ ] `schedule: List[Tuple[int, int, List[int]]]`
  - [ ] `parse_schedule(raw, default_ulysses, default_ring, world_size)`
- [ ] `parse_schedule` 支持以下输入：
  - [ ] `int`
  - [ ] `{degree: N}` / `{degree: N, ranks: [...]}`
  - [ ] `{ulysses: U, ring: R, ranks: [...]}`
- [ ] 增加 schedule 校验：
  - [ ] `degree == ulysses * ring`
  - [ ] `len(ranks) == degree`
  - [ ] ranks 不重复、无越界
- [ ] 实现 `DynamicSPManager`:
  - [ ] `_pre_create_meshes(device_type)`
  - [ ] `get_schedule_entry(step)`（循环索引）
  - [ ] `is_active(step)`
  - [ ] `get_broadcast_src(step)`
  - [ ] `sp_degree_changed(step)`
  - [ ] `apply_config(step)`（原地更新 `cp_config`）
  - [ ] `sync_output(output, hidden_states, step)`（广播 output）
  - [ ] `reset()`

**验收标准**
- schedule 能正确归一化并通过基础校验。
- manager 初始化时可预创建 mesh，运行时可按 step 切换配置。
- inactive rank 跳过 forward 后，仍能通过广播拿到一致输出。

---

## 2) 更新并行配置定义 `config.py`（P0）

- [ ] 修改 `src/cache_dit/parallelism/config.py`
- [ ] 将 `DynamicSPConfig` 定义迁移到 `dynamic_sp.py` 并在此导入使用
- [ ] 保留 `ParallelismConfig.dynamic_sp_config: Optional[DynamicSPConfig] = None`
- [ ] 清理潜在循环依赖（必要时使用局部导入）

**验收标准**
- `ParallelismConfig` 构造正常、类型引用正确。
- 代码中无重复 `DynamicSPConfig` 定义。

---

## 3) 更新配置加载 `load_configs.py`（P0）

- [ ] 修改 `src/cache_dit/caching/load_configs.py` 的 `load_parallelism_config`
- [ ] 从 `parallelism_config_kwargs` 提取 `dynamic_sp`
- [ ] 调用 `DynamicSPConfig.parse_schedule(...)` 构建配置
- [ ] 将 `dynamic_sp_config` 注入 `ParallelismConfig(...)`

**验收标准**
- YAML 中 `dynamic_sp.enabled/schedule` 可正确加载。
- 未配置 `dynamic_sp` 时保持现有行为不变。

---

## 4) 接入 CP 初始化流程（P0）

- [ ] 修改 `src/cache_dit/parallelism/transformers/context_parallelism/__init__.py`
- [ ] 在 `_enable_context_parallelism_ext(...)` 之后创建 `DynamicSPManager`
- [ ] 挂载 `transformer._dynamic_sp_manager = manager`
- [ ] 实现 `_wrap_forward_with_dynamic_sp(transformer, manager)`：
  - [ ] active ranks: `apply_config(step)` + 调用原始 forward
  - [ ] inactive ranks: 跳过 forward
  - [ ] all ranks: `sync_output(...)` 对齐输出
  - [ ] step 递增并按 schedule 循环

**验收标准**
- wrapper 在 CP hooks 外层，inactive rank 不进入 CP split/gather。
- 每步结束后所有 rank 输出一致，可继续执行 scheduler step。

---

## 5) 更新导出 `parallelism/__init__.py`（P1）

- [ ] 修改 `src/cache_dit/parallelism/__init__.py`
- [ ] 导出 `DynamicSPConfig` 与 `DynamicSPManager`

**验收标准**
- 外部可通过 `cache_dit.parallelism` 直接导入相关类。

---

## 6) 增加示例配置（P1）

- [ ] 新建 `examples/configs/dynamic_sp.yaml`
- [ ] 覆盖三类 schedule 写法：
  - [ ] `int`
  - [ ] `{degree, ranks}`
  - [ ] `{ulysses, ring, ranks}`

**验收标准**
- 示例配置可被 `load_parallelism_config` 直接解析。

---

## 7) 测试与回归验证（P0）

- [ ] 配置解析测试：
  - [ ] 合法配置
  - [ ] 非法配置（空、越界、重复、长度不匹配）
- [ ] 功能测试：8 卡 schedule 循环切换（如 8 -> 2 -> 4）
- [ ] 一致性测试：广播后所有 rank 输出一致（shape/dtype/value）
- [ ] 回归测试：`dynamic_sp.enabled=false` 时行为与现状一致
- [ ] 稳定性测试：步数超过 schedule 长度时循环行为正确

**验收标准**
- 关键测试用例通过，无新增 lint/type 错误，默认路径无回归。

---

## 8) 交付与文档（P1）

- [ ] 补充实现说明（模块职责、调用链、关键状态）
- [ ] 补充配置使用说明（如何开启、字段语义、常见错误）
- [ ] 记录风险与边界：
  - [ ] inactive rank 跳过 forward 的状态更新副作用
  - [ ] mesh padding 在特殊 world_size/degree 组合下的行为

**验收标准**
- 其他开发者可基于文档复现并验证 dynamic SP 行为。

---

## 建议排期

- Day 1: 完成 1 ~ 4（核心实现 + 接入）
- Day 2: 完成 5 ~ 8（导出、示例、测试、文档）
