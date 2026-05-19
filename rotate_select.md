# Group-wise Rotate Selection (MXFP4 / NVFP4)

## 1. 功能概览(test)

当前实现支持在量化时按 `w_group_size` 对每个 group 列块搜索旋转，并在每个 Transformer block 的四个输入槽位分别搜索：

- `qkv`（q/k/v 共享）
- `o`
- `gate_up`（gate/up 共享）
- `down`

开关参数：`--transform_search`

## 2. 旋转搜索目标（可选）

可选目标：

1. `mse`
- `L(T) = MSE(Q(W') - W')`

2. `cov`
- `L(T) = Tr(DeltaW * Cov' * DeltaW^T) / out_features`
- `W' = W T^{-T}`
- `Cov' = T^T Cov T`

3. `act_mse`
- `L(T) = E[ ||Q(xT) - xT||_2^2 ]`
- 直接在校准激活上评估旋转后激活的量化重构误差
- 当前实现要求：
  - `--a_bits < 16`
  - `--a_granularity=group`
  - `--a_group_size == --w_group_size`
- 搜索时会为每个槽位缓存最多 `--transform_search_act_sample_size` 行输入样本

4. `jtail`
- `J_tail(T) = L(T) + lambda * L_tail(T)`
- 其中 `L(T)` 由 `--transform_search_base_loss` 指定（`mse` / `cov` / `act_mse`）
- `L_tail(T)` 为分位点 bin 加权误差项
- `L_tail(T)` 的来源由 `--transform_search_tail_source {weight,activation}` 指定
- 当前合法组合：
  - `jtail(base=mse, tail=weight)`
  - `jtail(base=cov, tail=weight)`
  - `jtail(base=cov, tail=activation)`
  - `jtail(base=act_mse, tail=activation)`
- 当前非法组合：
  - `jtail(base=mse, tail=activation)`
  - `jtail(base=act_mse, tail=weight)`

## 3. A/B/混合误差来源与权重策略

你课件中的三类可直接映射到 `jtail` 的尾部权重 profile：

1. A类（outlier 挤压 normal）
- 命令：`--transform_search_tail_weight_mode=a_low`
- 含义：低分位 bin 权重更高

2. B类（大值主导）
- 命令：`--transform_search_tail_weight_mode=b_high`
- 含义：高分位 bin 权重更高

3. 混合
- 均匀：`--transform_search_tail_weight_mode=mixed_uniform`
- 中间更高：`--transform_search_tail_weight_mode=mixed_middle`

4. 自动A/B/混合（推荐）
- 命令：`--transform_search_tail_weight_mode=auto_abm`
- 含义：每个 group block 自动分析分布并在 `{a_low, b_high, mixed_uniform}` 中选一个。
- 规则（当前实现）：
  - `tail_source=weight`：基于该 group block 的 `|W|` 分位点（q50/q90/q99）进行启发式判别
  - `tail_source=activation`：基于该 group block 的输入激活 `|x|` 分位点（q50/q90/q99）进行启发式判别
  - `auto_abm` 仅在 `--transform_search_objective=jtail` 时生效。

补充：`two_tail` 为兼容旧实验保留（两端都高）。

权重形状由 `--transform_search_tail_weight_power` 控制（>0）。

## 4. 计算粒度说明（关键）

当前 `mse/cov/jtail(base=mse|cov, tail=weight)/L_tail(weight)` 的计算粒度是：

- 对某个 `group_idx` 的列块 `W[:, start:end]`（形状 `[out_features, group_size]`）计算误差
- 在同一槽位共享的层上求和（例如 qkv 会对 q/k/v 同一 `group_idx` 累加）

因此它是“按 group 列块决策”，不是 `rows*cols/group_size` 的逐元素暴力搜索。

`act_mse` 的计算粒度略有不同：

- 仍然是按当前 `group_idx` 决策候选 `T`
- 但打分对象不再是 `W[:, start:end]`
- 而是该槽位输入激活的 `x[:, start:end]`
- 即比较 `Q(x_g T) - x_g T` 的 MSE

如果使用 `jtail(base=cov|act_mse, tail=activation)`，则：

- 基础项 `L(T)` 来自 `cov` 或 `act_mse`
- 尾部项 `L_tail(T)` 在同一 group 的 `x[:, start:end]` 上计算
- 对共享槽位（如 `qkv`）来说，activation-tail 每个候选只计算一次，不按 q/k/v 分三次重复累加

## 5. 路径默认行为（objective=auto）

- GPTQ 路径（`--gptq`）：`auto -> cov`
- RTN 路径（无 `--gptq`）：`auto -> mse`

当目标需要 `cov`（`cov` 或 `jtail+base=cov`）时，会自动收集 `group_covariances`。

当目标需要激活域打分（`act_mse`，或 `jtail+base=act_mse`，或 `jtail+tail_source=activation`）时，会自动收集每个槽位的输入样本。

## 6. 参数说明

- `--transform_search_objective {auto,mse,cov,act_mse,jtail}`
- `--transform_search_base_loss {mse,cov,act_mse}`（仅 `jtail` 使用）
- `--transform_search_tail_source {weight,activation}`（仅 `jtail` 使用，默认 `weight` 以兼容旧行为）
- `--transform_search_act_sample_size <int>`（`act_mse` 和 activation-tail 使用）
- `--transform_search_tail_lambda <float>`
- `--transform_search_tail_bins <int>`
- `--transform_search_tail_weight_mode {a_low,b_high,mixed_uniform,mixed_middle,two_tail,auto_abm}`
- `--transform_search_tail_weight_power <float>`

## 7. 终端日志

搜索日志会打印目标与协方差状态，例如：

```text
[transform_search] objective=J_tail(base=cov,tail=activation,lambda=0.2,bins=4,weights=auto_abm,power=2) | group_covariances=enabled(qkv=256,o=256,gate_up=256,down=896) | act_samples=enabled(qkv=1024,o=1024,gate_up=1024,down=1024) | qkv=... | tail_mode=a_low:120,b_high:45,mixed_uniform:91
```

如果目标涉及激活域打分（`act_mse` 或 activation-tail），日志中会打印 `act_samples=enabled(...)`，表示每个槽位实际缓存的激活样本行数。

GPTQ 层误差日志：

```text
[self_attn.q_proj]: Relative Hessian error: 1.23e-03
```

## 8. 命令示例

### 8.1 GPTQ + act_mse

```bash
python model_quant.py \
  --model_name_or_path=/cephfs/shared/model/llama-3-8b-instruct \
  --dataset_name_or_path=${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt \
  --num_sequences=128 --sequence_length=2048 --dtype=auto \
  --format=nvfp --w_bits=4 --a_bits=4 --w_granularity=group --a_granularity=group \
  --w_group_size=16 --a_group_size=16 --w_observer=mse \
  --gptq --quantization_order=default \
  --transform_search --transform_search_objective=act_mse \
  --transform_search_act_sample_size=1024 \
  --transform_search_candidates identity hadamard dct dst gsr householder \
  --export_quantized_model=pseudoquant --save_path=outputs/nvfp_gptq_act_mse \
  --fuse_global_scale --amp
```

### 8.2 GPTQ + J_tail（A类：低分位更高）

```bash
python model_quant.py \
  --model_name_or_path=/cephfs/shared/model/llama-3-8b-instruct \
  --dataset_name_or_path=${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt \
  --num_sequences=128 --sequence_length=2048 --dtype=auto \
  --format=nvfp --w_bits=4 --a_bits=4 --w_granularity=group --a_granularity=group \
  --w_group_size=16 --a_group_size=16 --w_observer=mse \
  --gptq --quantization_order=default \
  --transform_search --transform_search_objective=jtail --transform_search_base_loss=cov \
  --transform_search_tail_lambda=0.2 --transform_search_tail_bins=4 \
  --transform_search_tail_weight_mode=a_low --transform_search_tail_weight_power=2.0 \
  --transform_search_candidates identity hadamard dct dst gsr householder \
  --export_quantized_model=pseudoquant --save_path=outputs/nvfp_gptq_jtail_a_low \
  --fuse_global_scale --amp
```

### 8.3 GPTQ + J_tail（B类：高分位更高）

```bash
python model_quant.py \
  --model_name_or_path=/cephfs/shared/model/llama-3-8b-instruct \
  --dataset_name_or_path=${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt \
  --num_sequences=128 --sequence_length=2048 --dtype=auto \
  --format=mxfp --w_bits=4 --a_bits=4 --w_granularity=group --a_granularity=group \
  --w_group_size=32 --a_group_size=32 --w_observer=mse \
  --gptq --quantization_order=default \
  --transform_search --transform_search_objective=jtail --transform_search_base_loss=cov \
  --transform_search_tail_lambda=0.2 --transform_search_tail_bins=4 \
  --transform_search_tail_weight_mode=b_high --transform_search_tail_weight_power=2.0 \
  --transform_search_candidates identity hadamard dct dst gsr householder \
  --export_quantized_model=pseudoquant --save_path=outputs/mxfp_gptq_jtail_b_high \
  --fuse_global_scale --amp
```

### 8.4 RTN + J_tail（混合：均匀）

```bash
python model_quant.py \
  --model_name_or_path=/cephfs/shared/model/llama-3-8b-instruct \
  --dataset_name_or_path=${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt \
  --num_sequences=128 --sequence_length=2048 --dtype=auto \
  --format=nvfp --w_bits=4 --a_bits=4 --w_granularity=group --a_granularity=group \
  --w_group_size=16 --a_group_size=16 --w_observer=minmax \
  --transform_search --transform_search_objective=jtail --transform_search_base_loss=mse \
  --transform_search_tail_lambda=0.1 --transform_search_tail_bins=4 \
  --transform_search_tail_weight_mode=mixed_uniform --transform_search_tail_weight_power=2.0 \
  --transform_search_candidates identity hadamard dct dst gsr householder \
  --export_quantized_model=pseudoquant --save_path=outputs/nvfp_rtn_jtail_mixed_uniform \
  --fuse_global_scale --amp
```

### 8.5 GPTQ + J_tail（自动A/B/混合）

```bash
python model_quant.py \
  --model_name_or_path=/cephfs/shared/model/llama-3-8b-instruct \
  --dataset_name_or_path=${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt \
  --num_sequences=128 --sequence_length=2048 --dtype=auto \
  --format=nvfp --w_bits=4 --a_bits=4 --w_granularity=group --a_granularity=group \
  --w_group_size=16 --a_group_size=16 --w_observer=mse \
  --gptq --quantization_order=default \
  --transform_search --transform_search_objective=jtail --transform_search_base_loss=cov \
  --transform_search_tail_lambda=0.2 --transform_search_tail_bins=4 \
  --transform_search_tail_weight_mode=auto_abm --transform_search_tail_weight_power=2.0 \
  --transform_search_candidates identity hadamard dct dst gsr householder \
  --export_quantized_model=pseudoquant --save_path=outputs/nvfp_gptq_jtail_auto_abm \
  --fuse_global_scale --amp
```

### 8.6 GPTQ + J_tail（cov base + activation tail）

```bash
python model_quant.py \
  --model_name_or_path=/cephfs/shared/model/llama-3-8b-instruct \
  --dataset_name_or_path=${OUT_CALIB}/fineweb_calib_1024x2048_tokens.pt \
  --num_sequences=128 --sequence_length=2048 --dtype=auto \
  --format=nvfp --w_bits=4 --a_bits=4 --w_granularity=group --a_granularity=group \
  --w_group_size=16 --a_group_size=16 --w_observer=mse \
  --gptq --quantization_order=default \
  --transform_search --transform_search_objective=jtail --transform_search_base_loss=cov \
  --transform_search_tail_source=activation \
  --transform_search_tail_lambda=0.2 --transform_search_tail_bins=4 \
  --transform_search_tail_weight_mode=auto_abm --transform_search_tail_weight_power=2.0 \
  --transform_search_act_sample_size=1024 \
  --transform_search_candidates identity hadamard dct dst gsr householder \
  --export_quantized_model=pseudoquant --save_path=outputs/nvfp_gptq_jtail_cov_acttail \
  --fuse_global_scale --amp
```
## 9. 关于 scale 搜索

scale 的求解机制未改，仍由 observer 决定：

- `--w_observer=minmax`：min-max
- `--w_observer=mse`：误差驱动 scale 搜索（实现中为 `|x-x_q|^2.4`）

也就是说，新增的是“旋转搜索目标与尾部权重策略”，不是把 scale 搜索改成 COV。






## 10. 旋转候选策略详解（数学原理 -> 效果 -> 场景）

先给统一背景：对某个 group 的候选旋转矩阵 `T`，实现中使用

- 激活变换：`x' = xT`
- 权重补偿：`W' = WT^{-T}`

在该约定下，线性层保持等价：

```text
x' (W')^T = (xT)(WT^{-T})^T = xW^T
```

搜索时就是在候选集合中比较不同 `T` 对量化误差目标（`mse/cov/jtail`）的影响。

### 10.1 `identity`

- 数学形式：`T = I`
- 性质：无旋转、无额外混合，`T^{-T}=I`
- 效果：
  - 不引入额外基变换误差
  - 当原权重分布已经“量化友好”时通常表现稳健
- 适用场景：
  - baseline 对照
  - 某些 group 已较均匀、无明显相关结构时

### 10.2 `hadamard`

- 数学形式（归一化 Hadamard）：
  - `T = H_g / sqrt(g)`，`H_g` 元素为 `±1`
  - `TT^T = I`，且 `T^{-1}=T^T=T`（对称正交）
- 代码实现要点：
  - 使用快速 Hadamard 变换（FHT），分组块处理
- 效果：
  - 强混合、能量扩散，常用于缓解少量通道主导（outlier 主导）问题
  - 工程上通常是最稳健候选之一
- 适用场景：
  - FP4 group 量化的默认强基线
  - 对“少数维度幅值过大”的 group 往往有效

### 10.3 `dct`（DCT-II, orthonormal）

- 数学形式（正交余弦基）：
  - `T = C_g`，`C_g C_g^T = I`
  - 元素可写为 `cos` 基（type-II, `norm='ortho'`）
- 代码实现要点：
  - 先构建 `group_size` 的 DCT 基块，再拼 block-diagonal
- 效果：
  - 去相关能力较强，对“平滑/低频相关”结构常更友好
  - 相比 Hadamard，混合更“频率有序”
- 适用场景：
  - 误差主要来自相关性而非极端离群值时
  - 某些 MLP/attention group 中可能优于 hadamard

### 10.4 `dst`（DST-II, orthonormal）

- 数学形式（正交正弦基）：
  - `T = S_g`，`S_g S_g^T = I`
  - 元素为 `sin` 基（type-II, `norm='ortho'`）
- 代码实现要点：
  - 与 DCT 类似，按 group 基块拼 block-diagonal
- 效果：
  - 与 DCT 互补，偏向不同边界/频率结构
  - 在部分 group 上会比 DCT/Hadamard 更优（取决于分布）
- 适用场景：
  - 数据在某些 group 呈现与正弦基更匹配的结构时
  - 作为 DCT 的互补候选很有价值

### 10.5 `gsr`（Hadamard 重排序基）

- 数学形式（按列重排的 Hadamard）：
  - 先取 Hadamard 基 `H_g`
  - 统计每列符号翻转次数并排序，得到置换矩阵 `P`
  - `T = (H_g P) / sqrt(g)`
- 性质：
  - 仍是正交变换（仅列置换 + 归一化）
- 效果：
  - 相当于“按 sequency 重排”的 Hadamard 混合
  - 有时可在 Hadamard 与频率基之间取得更好折中
- 适用场景：
  - 需要保留 Hadamard 的高效/稳定性，但希望基顺序更匹配数据

### 10.6 `householder`

- 数学形式（per-group data-adaptive 单反射）：
  - `T_g = I - 2v_gv_g^T/(v_g^Tv_g)`
  - `v_g = a_g - b`，其中 `a_g` 是当前 group 的归一化能量方向，`b` 是均匀方向
- 性质：
  - 每个 group 独立构造自己的 `T_g`
  - 正交、且自逆：`T_g^{-1}=T_g`，`T_g^T=T_g`
  - 参数自由度低（由一个向量确定）
- 效果：
  - 以“反射”方式将局部能量集中方向映射到更均匀方向
  - 相比旧版随机共享 Householder，更符合 group-wise 自适应搜索
  - 在很多日志里是 identity 之后的高频次候选之一
- 适用场景：
  - group 内存在明显主方向/偏置时
  - 需要低复杂度但非平凡的正交变换
- 数据来源：
  - `mse`：基于当前 group 的权重 RMS 能量
  - `cov`：基于当前 group 的激活协方差对角线能量
  - `act_mse`：基于当前 group 的激活样本 RMS 能量
  - `jtail`：`tail_source=weight` 时基于权重，`tail_source=activation` 时基于激活

### 10.7 `fast_food`（可选，不在默认候选内）

- 数学形式（结构化随机变换）：
  - `T = (1/sigma)(1/sqrt(g)) S H G P H B`
  - 其中 `B/G/S` 为对角随机矩阵，`P` 为置换，`H` 为 Hadamard
- 性质：
  - 强随机混合，结构化且可高效实现
- 效果：
  - 潜在表达能力强，但稳定性/依赖环境敏感，通常用于扩展实验
- 适用场景：
  - 研究型探索、候选池扩展（`--transform_search_candidates ... fast_food`）

### 10.8 候选策略对比（汇报可直接引用）

| 候选 | 数学性质 | 典型效果 | 常见适用场景 |
|---|---|---|---|
| `identity` | `T=I` | 最稳健基线，不做混合 | 已量化友好的 group、对照实验 |
| `hadamard` | 正交、自逆、强混合 | 能量扩散，常缓解 outlier 主导 | FP4 group 默认强基线 |
| `dct` | 正交余弦基 | 去相关/频率有序，常对平滑结构有利 | 相关性主导误差 |
| `dst` | 正交正弦基 | 与 DCT 互补，适配不同边界/频率结构 | 与 DCT 互补探索 |
| `gsr` | Hadamard 列重排后仍正交 | Hadamard 稳定性 + 基顺序调优 | 需在稳定与匹配间折中 |
| `householder` | per-group 单反射，正交自逆 | 低成本自适应方向重定向 | 主方向/能量集中明显的 group |
| `fast_food`* | 结构化随机复合变换 | 混合能力强，实验性更强 | 扩展候选池研究 |

`*` 默认候选不包含 `fast_food`，需显式加入。

### 10.9 结果解读建议

同一层不同 group 被选中的候选可不同，这是设计目标。日志中例如：

```text
[transform_search] ... | qkv=identity:119, householder:71, dst:28, hadamard:17, dct:15, gsr:6
```

表示该槽位下各 group 的“最优候选计数”。计数越高仅表示在当前目标函数与校准集上更常获胜，不代表全局绝对最优。
