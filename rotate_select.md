# Group-wise Rotate Selection (MXFP4 / NVFP4)

## 1. 功能概览

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

3. `jtail`
- `J_tail(T) = L(T) + lambda * L_tail(T)`
- 其中 `L(T)` 由 `--transform_search_base_loss` 指定（`mse` 或 `cov`）
- `L_tail(T)` 为分位点 bin 加权误差项

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
- 规则（当前实现）：基于该 group block 的 `|W|` 分位点（q50/q90/q99）进行启发式判别。`auto_abm` 仅在 `--transform_search_objective=jtail` 时生效。

补充：`two_tail` 为兼容旧实验保留（两端都高）。

权重形状由 `--transform_search_tail_weight_power` 控制（>0）。

## 4. 计算粒度说明（关键）

当前 `mse/cov/jtail/L_tail` 的计算粒度是：

- 对某个 `group_idx` 的列块 `W[:, start:end]`（形状 `[out_features, group_size]`）计算误差
- 在同一槽位共享的层上求和（例如 qkv 会对 q/k/v 同一 `group_idx` 累加）

因此它是“按 group 列块决策”，不是 `rows*cols/group_size` 的逐元素暴力搜索。

## 5. 路径默认行为（objective=auto）

- GPTQ 路径（`--gptq`）：`auto -> cov`
- RTN 路径（无 `--gptq`）：`auto -> mse`

当目标需要 `cov`（`cov` 或 `jtail+base=cov`）时，会自动收集 `group_covariances`。

## 6. 参数说明

- `--transform_search_objective {auto,mse,cov,jtail}`
- `--transform_search_base_loss {mse,cov}`（仅 `jtail` 使用）
- `--transform_search_tail_lambda <float>`
- `--transform_search_tail_bins <int>`
- `--transform_search_tail_weight_mode {a_low,b_high,mixed_uniform,mixed_middle,two_tail,auto_abm}`
- `--transform_search_tail_weight_power <float>`

## 7. 终端日志

搜索日志会打印目标与协方差状态，例如：

```text
[transform_search] objective=J_tail(base=cov,lambda=0.2,bins=4,weights=auto_abm,power=2) | group_covariances=enabled(qkv=256,o=256,gate_up=256,down=896) | qkv=... | tail_mode=a_low:120,b_high:45,mixed_uniform:91
```

GPTQ 层误差日志：

```text
[self_attn.q_proj]: Relative Hessian error: 1.23e-03
```

## 8. 命令示例

### 8.1 GPTQ + J_tail（A类：低分位更高）

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

### 8.2 GPTQ + J_tail（B类：高分位更高）

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

### 8.3 RTN + J_tail（混合：均匀）

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

### 8.4 GPTQ + J_tail（自动A/B/混合）

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
## 9. 关于 scale 搜索

scale 的求解机制未改，仍由 observer 决定：

- `--w_observer=minmax`：min-max
- `--w_observer=mse`：误差驱动 scale 搜索（实现中为 `|x-x_q|^2.4`）

也就是说，新增的是“旋转搜索目标与尾部权重策略”，不是把 scale 搜索改成 COV。





