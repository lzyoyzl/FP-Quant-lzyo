# LM-Eval Summary (BoolQ / ARC / PIQA / WinoGrande)

**显示格式**：单元格内尽量同时展示 `acc` 与 `acc_norm`（若存在）。

**平均规则（严格按任务）**：
- BoolQ：用 `acc`
- ARC-Easy：用 `acc_norm`（若存在同时展示 `acc`；平均只取 `acc_norm`）
- ARC-Challenge：用 `acc_norm`（同上）
- PIQA：用 `acc_norm`（同上）
- WinoGrande：用 `acc`

> 数值以百分数显示（例如 73.25 表示 73.25%）。Avg 行括号内 n 表示参与平均的任务数。

### BASE

| Task | llama-2-7b-hf | llama-3-8b-hf | Qwen3-8B |
| --- | --- | --- | --- |
| boolq | acc=79.36 | acc=82.14 | acc=86.61 |
| arc_easy | acc_norm=73.86 (acc=75.63) | acc_norm=77.53 (acc=80.72) | acc_norm=80.93 (acc=83.42) |
| arc_challenge | acc_norm=52.56 (acc=49.57) | acc_norm=58.19 (acc=55.72) | acc_norm=67.24 (acc=64.93) |
| piqa | acc_norm=78.40 (acc=78.18) | acc_norm=80.58 (acc=78.78) | acc_norm=77.80 (acc=76.61) |
| winogrande | acc=74.35 | acc=78.22 | acc=70.56 |
| Avg (task-wise primary) | 71.71 (n=5) | 75.33 (n=5) | 76.63 (n=5) |

### mxfp_gptq_w4a4_hadamard_h128_mse_default

| Task | llama-2-7b-hf | llama-3-8b-hf | Qwen3-8B |
| --- | --- | --- | --- |
| boolq | acc=77.34 | acc=79.08 | acc=85.72 |
| arc_easy | acc_norm=73.53 (acc=75.34) | acc_norm=75.51 (acc=77.31) | acc_norm=78.79 (acc=80.43) |
| arc_challenge | acc_norm=51.88 (acc=48.55) | acc_norm=54.61 (acc=51.62) | acc_norm=64.25 (acc=61.26) |
| piqa | acc_norm=78.13 (acc=77.04) | acc_norm=77.80 (acc=76.66) | acc_norm=75.90 (acc=75.95) |
| winogrande | acc=71.98 | acc=73.24 | acc=68.19 |
| Avg (task-wise primary) | 70.57 (n=5) | 72.05 (n=5) | 74.57 (n=5) |

### mxfp_mrgptq_w4a4_hadamard_h128_mse_activation

| Task | llama-2-7b-hf | llama-3-8b-hf | Qwen3-8B |
| --- | --- | --- | --- |
| boolq | acc=75.75 | acc=79.57 | acc=85.47 |
| arc_easy | acc_norm=72.31 (acc=75.17) | acc_norm=76.64 (acc=77.95) | acc_norm=79.12 (acc=82.03) |
| arc_challenge | acc_norm=50.43 (acc=47.01) | acc_norm=53.75 (acc=49.91) | acc_norm=65.36 (acc=62.46) |
| piqa | acc_norm=77.31 (acc=77.20) | acc_norm=78.51 (acc=77.53) | acc_norm=76.39 (acc=75.63) |
| winogrande | acc=71.67 | acc=72.45 | acc=69.06 |
| Avg (task-wise primary) | 69.49 (n=5) | 72.19 (n=5) | 75.08 (n=5) |
