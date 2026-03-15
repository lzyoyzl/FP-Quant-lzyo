# Group-wise Rotate Selection (MXFP4 / NVFP4)

## 1. 功能概述

当前实现支持在量化时按 `group` 自动选择最优旋转算法（GPTQ 路径默认目标：GPTQ 一致的激活加权误差；RTN 路径回退为量化 MSE）。

- 开关参数：`--transform_search`
- 候选旋转：`--transform_search_candidates`
- 默认候选：`identity hadamard dct dst gsr householder`
- 搜索粒度：按 `w_group_size` 分组，在每个 group 上独立选最优旋转
- 作用位置：每个 block 的 `qkv / o / gate_up / down` 四类输入变换分别搜索

## 1.1 当前搜索目标（GPTQ 一致）

当启用 `--gptq --transform_search` 时，当前实现不再使用纯 weight-MSE 作为唯一目标，而是使用 GPTQ 一致的局部加权误差：

- 对每个 group、每个候选变换 `T`：
  - 权重旋转：`W' = W T^{-T}`
  - 量化误差：`DeltaW = Q(W') - W'`
  - 激活协方差旋转：`Cov' = T^T Cov T`
  - 打分：`score = Tr(DeltaW * Cov' * DeltaW^T) / out_features`

说明：

- `Cov` 来自该 block 校准数据上对应槽位输入（qkv/o/gate_up/down）的 group 协方差统计。
- 该目标与 GPTQ 的 Hessian 加权思想一致，比纯 MSE 更贴近最终量化目标。
- 若协方差不可用（如 RTN 路径），会自动回退到原有 MSE 目标。

## 2. 线性等价性保证

实现中使用的是等价配对：

- 激活：`x' = xT`
- 权重：`W' = W T^{-T}`

并且代码里对分组混合旋转增加了严格检查，确保 `T @ (T^{-T})^T ≈ I`。

## 3. 使用建议

- `--transform_search` 主要用于 weight 量化（要求 `w_bits < 16` 且 `w_granularity=group`）
- 导出模型时，建议 `w_bits=4` 且 `a_bits=4`
- `nvfp` 会自动修正 group size 到 16、scale precision 到 `e4m3`
- `mxfp` 会自动修正 group size 到 32、scale precision 到 `e8m0`

## 3.1 旋转方法对比表（含示例与效果）

> 说明：下表“效果”是 FP4 group 量化中的常见现象，不是绝对结论；最终以 `--transform_search` 实测 MSE 为准。

| 方法 | 原理（每个 group 长度为 g） | 最小示例 | 常见效果（对量化 MSE） | 代价与约束 |
|---|---|---|---|---|
| `identity` | `T = I`，不做特征混合。 | `g=4`，`x=[x1,x2,x3,x4]`，则 `x'=x`。 | 作为基线最稳定，但通常没有额外降误差收益。 | 开销最低，无额外约束。 |
| `hadamard` | `T = H_g / sqrt(g)`，通过 `+1/-1` 正交基做全维度混合。 | `g=4` 时 `T = 1/2 * [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]`。 | 对“局部极值明显/能量集中”的 group 通常有效，MSE 常明显下降。 | 开销低；工程上通常更适合 `g` 为 2 的幂。 |
| `dct` | DCT-II 正交余弦基，把信号投影到“低频到高频”坐标。 | `g=4` 块矩阵可由 `dct(I4, type=2, norm='ortho')` 构造。 | 对平滑或相关性强的 group 往往有收益；有些层会优于 hadamard。 | 需构造稠密矩阵，开销中等。 |
| `dst` | DST-II 正交正弦基，与 DCT 的边界特性不同。 | `g=4` 块矩阵可由 `dst(I4, type=2, norm='ortho')` 构造。 | 在部分分布下会优于 DCT；整体表现依赖具体层与 group。 | 与 DCT 类似，开销中等。 |
| `gsr` | 基于 Hadamard，并按列的 sequency（符号变化次数）排序重排。 | `g=4` 时，列按变化次数排序为 `[col1, col3, col4, col2]`，再做 `1/sqrt(g)` 归一化。 | 在有模式结构的 group 上常比纯 Hadamard 更稳，但不保证全局最优。 | 开销低到中等；依赖 Hadamard 基构造。 |
| `householder` | Householder 反射：`T = I - 2vv^T/(v^Tv)`，属于正交变换且 `T^{-1}=T^T=T`。 | `g=4`，取 `v=[1,2,0,0]`，可得一个 4x4 反射矩阵并对该组整体镜像反射。 | 常能在部分层上降低 MSE，尤其对方向性较强的 group；效果通常介于 `identity` 与 `hadamard/dct` 之间，需搜索判定。 | 构造开销低，矩阵稠密；需要 `v` 非零（实现中随机生成并做数值保护）。 |
## 4. 量化 + 导出命令

### 4.1 RTN + NVFP + pseudoquant 导出

```powershell
python model_quant.py --model_name_or_path=Qwen/Qwen3-8B --dataset_name_or_path=fineweb-edu --num_sequences=128 --sequence_length=2048 --format=nvfp --w_bits=4 --a_bits=4 --w_granularity=group --a_granularity=group --w_observer=minmax --transform_class=identity --hadamard_group_size=128 --transform_search --transform_search_candidates identity hadamard dct dst gsr householder --export_quantized_model=pseudoquant --save_path=quantized_models/qwen3-8b-nvfp-rtn-rotate-select --cpu_offload_modules --cpu_offload_activations --fuse_global_scale --amp
```

#### 4.1.1 参数解释（与上面命令一一对应）

- `python model_quant.py`：运行主量化脚本。
- `--model_name_or_path=Qwen/Qwen3-8B`：待量化基座模型。
- `--dataset_name_or_path=fineweb-edu`：校准数据集名称或路径。
- `--num_sequences=128`：用于校准的样本条数。
- `--sequence_length=2048`：每条校准样本的 token 长度。
- `--format=nvfp`：使用 NVFP4 路径。
- `--w_bits=4`：权重量化到 4bit。
- `--a_bits=4`：激活量化到 4bit。
- `--w_granularity=group`：权重按 group 统计量化参数（transform_search 必需）。
- `--a_granularity=group`：激活按 group 统计量化参数。
- `--w_observer=minmax`：权重量化参数观测器为 min-max。
- `--transform_class=identity`：基础 transform 选项；在 `--transform_search` 开启时不参与搜索决策，仅作兼容保留。
- `--hadamard_group_size=128`：传统 transform 的分组参数；在当前 per-group 搜索中，搜索粒度由 `w_group_size` 决定（NVFP 下自动为 16）。
- `--transform_search`：开启按 group 搜索旋转算法（GPTQ 路径使用 GPTQ 一致目标；RTN 路径回退到 MSE 目标）。
- `--transform_search_candidates identity hadamard dct dst gsr householder`：搜索候选算法集合（已包含 householder）。
- `--export_quantized_model=pseudoquant`：导出 pseudoquant 权重。
- `--save_path=...`：导出目录。
- `--cpu_offload_modules`：模块计算阶段可在 CPU/GPU 间搬运，降低显存占用。
- `--cpu_offload_activations`：中间激活可下放 CPU，进一步省显存。
- `--fuse_global_scale`：融合 qkv 与 gate/up 的全局 scale（该项在 NVFP 的 `e4m3` 路径有效）。
- `--amp`：启用 autocast 混合精度以减少显存并提速部分步骤。

### 4.2 GPTQ + MXFP + realquant 导出

```powershell
python model_quant.py --model_name_or_path=meta-llama/Llama-3.1-8B-Instruct --dataset_name_or_path=fineweb-edu --num_sequences=128 --sequence_length=2048 --format=mxfp --w_bits=4 --a_bits=4 --w_granularity=group --a_granularity=group --w_observer=minmax --gptq --quantization_order=default --rel_damp=1e-2 --transform_class=identity --hadamard_group_size=128 --transform_search --transform_search_candidates identity hadamard dct dst gsr householder --export_quantized_model=realquant --save_path=quantized_models/llama31-8b-mxfp-gptq-rotate-select --cpu_offload_modules --cpu_offload_activations --fuse_global_scale --amp
```

#### 4.2.1 参数解释（与上面命令一一对应）

- `python model_quant.py`：运行主量化脚本。
- `--model_name_or_path=meta-llama/Llama-3.1-8B-Instruct`：待量化基座模型。
- `--dataset_name_or_path=fineweb-edu`：校准数据集名称或路径。
- `--num_sequences=128`：用于校准的样本条数。
- `--sequence_length=2048`：每条校准样本长度。
- `--format=mxfp`：使用 MXFP4 路径。
- `--w_bits=4`：权重量化到 4bit。
- `--a_bits=4`：激活量化到 4bit。
- `--w_granularity=group`：权重按 group 统计量化参数（transform_search 必需）。
- `--a_granularity=group`：激活按 group 统计量化参数。
- `--w_observer=minmax`：权重量化参数观测器为 min-max。
- `--gptq`：使用 GPTQ 量化流程（不加该参数则走 RTN）。
- `--quantization_order=default`：GPTQ 列处理顺序为默认顺序。
- `--rel_damp=1e-2`：GPTQ Hessian 正则阻尼系数。
- `--transform_class=identity`：基础 transform 选项；在 `--transform_search` 开启时不参与搜索决策，仅作兼容保留。
- `--hadamard_group_size=128`：传统 transform 分组参数；当前 per-group 搜索时，搜索粒度由 `w_group_size` 决定（MXFP 下自动为 32）。
- `--transform_search`：开启按 group 搜索旋转算法（在 GPTQ 路径下使用 GPTQ 一致目标）。
- `--transform_search_candidates identity hadamard dct dst gsr householder`：候选旋转算法（已包含 householder）。
- `--export_quantized_model=realquant`：导出 realquant 权重。
- `--save_path=...`：导出目录。
- `--cpu_offload_modules`：模块 offload 以降低显存峰值。
- `--cpu_offload_activations`：激活 offload 以降低显存峰值。
- `--fuse_global_scale`：该参数仅在 `scale_precision=e4m3`（NVFP）分支会生效；对本命令（MXFP -> `e8m0`）通常无实际作用。
- `--amp`：启用 autocast 混合精度。

## 5. 命令隐式行为（由脚本自动修正）

- 当 `--format=nvfp`：
  - `w_group_size` 自动改为 `16`
  - `a_group_size` 自动改为 `16`
  - `scale_precision` 自动改为 `e4m3`
- 当 `--format=mxfp`：
  - `w_group_size` 自动改为 `32`
  - `a_group_size` 自动改为 `32`
  - `scale_precision` 自动改为 `e8m0`

## 6. 常见问题

- 如果某个候选旋转依赖不可用或矩阵不可逆，会在搜索时自动跳过。
- 如果未安装依赖（如 `safetensors` / `transformers`），请先安装后再运行导出。



