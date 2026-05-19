# 旋转搜索实现汇报（FP-Quant 项目）

## 1. 背景与目标

本次工作在原有固定旋转（`identity/hadamard`）基础上，新增了**按 group 自动选择旋转变换**的能力，目标是降低量化误差并支持 NVFP4/MXFP4 的统一实验流程。

核心目标：

1. 对每个输入分组（group block）独立搜索最优旋转，而不是全层只用一种旋转。
2. 支持多候选变换并自动选择（`identity/hadamard/dct/dst/gsr/householder`，可扩展 `fast_food`），其中 `householder` 为 per-group data-adaptive 候选。
3. 搜索目标可配置：`mse`、`cov`（GPTQ 一致）、`act_mse`（激活量化重构误差）、`jtail`（尾部分位点加权目标）。
4. 保证线性层前后等价（在浮点域）并可导出到 pseudoquant/realquant。

---

## 2. 总体设计

### 2.1 分槽位（slot）搜索

以 Transformer block 为单位，对以下 4 个输入槽位分别搜索输入变换：

1. `q/k/v` 共享一个输入变换。
2. `o_proj` 一个输入变换。
3. `gate/up` 共享一个输入变换。
4. `down_proj` 一个输入变换。

对应实现入口：

- [`build_block_input_transforms`](src/quantization/transform_search.py)

这样做的原因：

1. 与模型结构一致（这些层共享同一输入空间或强相关输入空间）。
2. 搜索复杂度可控（相较于更细粒度按每个权重 group 独立建模）。

### 2.2 Group-wise 混合变换

新增 `MixedGroupTransform`，允许一个线性层不同 group 选不同旋转块（分块对角思想）。

关键点：

1. 存储 `forward_matrices/backward_matrices`（形状 `[num_groups, g, g]`）。
2. 前向通过 `einsum` 对每个 group 独立变换。
3. 增加线性等价校验：`forward @ backward^T ≈ I`。

对应实现：

- [`MixedGroupTransform`](src/quantization/transform_search.py)

### 2.3 线性等价保证

采用行向量约定：

- 激活：`x' = xT`
- 权重：`W' = WT^{-T}`

则 `x'W'^T = xW^T`。

代码层面，搜索时对权重 group 使用候选 `backward`（即 `T^{-T}`）旋转，协方差分支中使用 `Cov' = T^T Cov T`，并在 `MixedGroupTransform` 初始化时做一致性检查。

### 2.4 Per-group adaptive Householder

`householder` 候选已由旧版“全体 group 共享一个随机反射矩阵”改为“每个 group 独立构造一个数据自适应反射矩阵”。

构造方式：

1. 先从当前 group 得到通道 RMS 能量方向 `a_g`。
2. 令 `b` 为均匀方向。
3. 构造 `v_g = a_g - b`，并得到 `T_g = I - 2v_gv_g^T/(v_g^Tv_g)`。

数据来源随目标变化：

1. `mse`：权重 group 能量。
2. `cov`：激活协方差对角线能量。
3. `act_mse`：激活样本能量。
4. `jtail`：随 `tail_source` 选择权重或激活能量。

该设计保留 Householder 的正交、自逆、低成本特性，同时避免旧版随机共享反射方向与不同 group 局部分布不匹配的问题。

---

## 3. 搜索目标函数设计

### 3.1 支持目标

通过 CLI 参数控制：

1. `--transform_search_objective {auto,mse,cov,act_mse,jtail}`
2. `--transform_search_base_loss {mse,cov,act_mse}`（仅 `jtail` 使用）
3. `--transform_search_tail_source {weight,activation}`（仅 `jtail` 使用，默认 `weight` 以兼容旧行为）
4. `--transform_search_act_sample_size`（`act_mse` 与 activation-tail 使用）

对应实现：

- [`resolve_transform_search_config`](src/quantization/transform_search.py)
- [`should_collect_group_covariances`](src/quantization/transform_search.py)

### 3.2 `mse` 目标

对候选旋转后权重量化误差做均方误差评估，选最小者。

### 3.3 `cov` 目标（GPTQ 一致近似）

使用加权误差近似 GPTQ Hessian 目标：

- 局部项：`Tr(ΔW * Cov' * ΔW^T) / out_features`
- 其中 `Cov' = T^T Cov T`

对应实现：

- [`_compute_gptq_consistent_group_error`](src/quantization/transform_search.py)

### 3.4 `act_mse` 目标

直接在校准激活上评估旋转后的激活量化重构误差：

- `L_act(T) = E[ ||Q(xT) - xT||_2^2 ]`

实现要点：

1. 为每个槽位缓存有上限的输入样本行。
2. 对当前 `group_idx` 的输入块 `x_g` 计算 `x_g T`。
3. 用当前激活量化器配置直接量化 `x_g T`，并比较 `Q(x_g T)` 与 `x_g T` 的 MSE。

当前实现约束：

1. `a_bits < 16`
2. `a_granularity = group`
3. `a_group_size = w_group_size`

对应实现：

- [`collect_block_slot_input_samples`](src/quantization/gptq.py)
- [`search_best_group_transform`](src/quantization/transform_search.py)

### 3.5 `jtail` 目标

在基础目标 `L(T)` 上加尾部分位点项：

- `J(T) = L(T) + λ * L_tail(T)`

`L_tail` 通过分位点分桶和权重模式建模误差来源差异。

长期接口约束：

1. `jtail(base=mse, tail=weight)`：允许
2. `jtail(base=cov, tail=weight)`：允许
3. `jtail(base=cov, tail=activation)`：允许
4. `jtail(base=act_mse, tail=activation)`：允许
5. `jtail(base=mse, tail=activation)`：禁止
6. `jtail(base=act_mse, tail=weight)`：禁止

两类尾部项含义：

1. `tail_source=weight`
   - 按 `|W'|` 的分位点分桶
   - 对 `ΔW = Q(W') - W'` 的平方误差做 bin 加权
2. `tail_source=activation`
   - 按 `|xT|` 的分位点分桶
   - 对 `ΔX = Q(xT) - xT` 的平方误差做 bin 加权

其中 `base=act_mse` 与 `tail=activation` 共享同一组槽位输入样本；`base=cov` 与 `tail=activation` 则是“输出误差代理 + 激活尾部结构偏好”的混合目标。

支持权重模式：

1. `a_low`
2. `b_high`
3. `mixed_uniform`
4. `mixed_middle`
5. `two_tail`
6. `auto_abm`

`auto_abm` 会按 group block 分布自动选择 A/B/mixed：

1. `tail_source=weight`：基于 `|W|` 分位点启发式分类
2. `tail_source=activation`：基于 `|x|` 分位点启发式分类

对应实现：

- [`_build_tail_bin_weights`](src/quantization/transform_search.py)
- [`_select_tail_weight_mode_for_group`](src/quantization/transform_search.py)
- [`_compute_tail_quantile_error`](src/quantization/transform_search.py)

---

## 4. RTN 与 GPTQ 路径差异

### 4.1 默认 objective

1. RTN 路径：`auto -> mse`
2. GPTQ 路径：`auto -> cov`

对应调用：

- RTN：[`rtn_quantization`](src/quantization/rtn.py)
- GPTQ：[`gptq_quantization`](src/quantization/gptq.py)

### 4.2 协方差与激活样本收集

1. 仅在目标需要 `cov` 时收集 slot 协方差。
2. 在目标涉及激活域打分时收集 slot 输入样本：
   - `act_mse`
   - `jtail(base=act_mse, tail=activation)`
   - `jtail(base=cov, tail=activation)`

对应实现：

- [`collect_block_slot_input_covariances`](src/quantization/gptq.py)
- [`should_collect_group_covariances`](src/quantization/transform_search.py)
- [`collect_block_slot_input_samples`](src/quantization/gptq.py)
- [`should_collect_input_samples`](src/quantization/transform_search.py)

### 4.3 `quantization_order` 作用域

`--quantization_order` 仅在 GPTQ 生效（`default/activation`），RTN 不使用。

---

## 5. 导出与推理侧兼容设计

### 5.1 导出矩阵策略

新增 `get_export_transform_matrices`：

1. pseudoquant：支持导出 group-wise 全矩阵 bank（3D）。
2. realquant：回退为单矩阵（兼容旧后端约束）。

对应实现：

- [`get_export_transform_matrices`](src/quantization/transform_search.py)

### 5.2 RTN/GPTQ 导出接入

RTN 与 GPTQ 都改为从“实际选中 transform 实例”取矩阵，不再依赖单一 `transform_class` 常量矩阵。

对应实现：

- RTN 导出段：[`rtn.py`](src/quantization/rtn.py)
- GPTQ 导出段：[`gptq.py`](src/quantization/gptq.py)

---

## 6. 参数接口（新增）

在 `model_quant.py` 新增旋转搜索参数：

1. `--transform_search`
2. `--transform_search_candidates`
3. `--transform_search_objective`
4. `--transform_search_base_loss`
5. `--transform_search_tail_source`
6. `--transform_search_tail_lambda`
7. `--transform_search_tail_bins`
8. `--transform_search_tail_weight_mode`
9. `--transform_search_tail_weight_power`
10. `--transform_search_act_sample_size`

对应代码：

- [`model_quant.py`](model_quant.py)

---

## 7. 关键代码改动清单（按文件）

### 7.1 `src/quantization/transform_search.py`

完成内容：

1. 新增 `MixedGroupTransform`（group-wise 混合旋转）。
2. 新增候选集合与目标/权重模式校验。
3. 新增 `mse/cov/act_mse/jtail` 目标计算与解析。
4. 新增 `jtail(base_loss, tail_source)` 合法组合校验与 activation-tail 分支。
5. 新增 `auto_abm` 分布分类（按 tail_source 选择基于权重或激活）。
6. 新增 per-group data-adaptive `householder` 候选（按当前 group 的权重/协方差/激活能量构造）。
7. 新增 `search_best_group_transform`（逐 group 搜索）。
8. 新增 `build_block_input_transforms`（4 槽位整合）。
9. 新增 `format_transform_summary` 与 objective 输出辅助。
10. 新增 `get_export_transform_matrices`（导出兼容）。

### 7.2 `model_quant.py`

完成内容：

1. 增加 transform search 相关 CLI 参数。
2. 增加 `--transform_search_tail_source`，默认 `weight` 保持旧行为。
3. 与 RTN/GPTQ 路径联动。

### 7.3 `src/quantization/rtn.py`

完成内容：

1. 在 block 级调用搜索构建 4 个输入 transform。
2. 按 objective 决定是否收集协方差/激活样本。
3. 打印每个 block 的搜索摘要（候选命中统计、协方差状态、激活样本状态）。
4. 导出时使用 transform 实例矩阵（支持 group-wise pseudoquant）。

### 7.4 `src/quantization/gptq.py`

完成内容：

1. 同步引入 transform search 与 objective 选择。
2. 协方差收集、激活样本收集与 `cov/act_mse/jtail+activation-tail` 路径支持。
3. 导出矩阵逻辑与 RTN 对齐。
4. 日志标签统一为 `Relative Hessian error`（GPTQ阶段指标）。

### 7.5 `src/quantization/quantizer.py`

完成内容（稳定性修复）：

1. `MSE observer` 内部误差缓存改为 `float32`。
2. `quantization_error` 比较路径统一到 `float32`。
3. 修复 RTN + MSE 场景下的 Half/Float 赋值报错。

---

## 8. 当前已验证行为

1. 搜索日志可输出每层槽位命中分布（例如 `qkv=identity:..., householder:...`）。
2. RTN/GPTQ 均支持搜索流程，且 `auto` 默认随路径变化：RTN->mse，GPTQ->cov。
3. 已支持 per-group adaptive `householder` 候选并可参与搜索。
4. 已支持 `jtail`、`auto_abm`、`act_mse` 与 activation-tail 策略。

---

## 9. 已知注意事项

1. `--quantization_order` 仅 GPTQ 生效，RTN 传该参数不影响行为。
2. 如果搜索目标选 `cov` 或 `jtail+base=cov`，必须有协方差收集。
3. 如果搜索目标涉及激活域打分（`act_mse` 或 activation-tail），当前要求 `a_bits<16` 且 `a_group_size=w_group_size`。
4. `jtail(base=mse, tail=activation)` 与 `jtail(base=act_mse, tail=weight)` 当前明确禁止。
5. `realquant` 导出当前对 group-wise mixed transform 会退化为单矩阵（兼容性优先）。

---

## 10. 建议的汇报主线（可直接口述）

1. 问题：固定旋转无法适配不同 group 的误差来源。
2. 方案：按 group 搜索 + 按槽位共享，兼顾效果与复杂度。
3. 目标函数：从 MSE 扩展到 GPTQ 一致（cov）、激活重构驱动（act_mse）与分位点尾部增强（jtail）。
4. 等价性：通过 `x'=xT, W'=WT^{-T}` 与矩阵一致性检查保证。
5. 工程落地：RTN/GPTQ/导出三条链路全部接入，参数化可复现实验。
6. 风险与后续：realquant mixed transform 的后端支持可继续增强。
