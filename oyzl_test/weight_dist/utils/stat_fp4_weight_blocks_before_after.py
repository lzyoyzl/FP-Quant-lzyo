#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils:
python stat_fp4_weight_blocks_before_after.py \
  --before_model_dir /cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant/llama-3-8b-instruct/mxfp_gptq_w4a4_identity_mse_default \
  --after_model_dir  /cephfs/shared/zlouyang/FP-Quant/oyzl_test/results_quant/llama-3-8b-instruct/mxfp_gptq_w4a4_hadamard_h128_mse_default \
  --out_csv ./fp4_block_stats_llama3_mxfp_gptq.csv \
  --torch_threads 16



统计 pseudoquant 导出模型中 dqweight 的权重分布，并按 hadamard_group_size 沿列分块。
支持对比两份模型：
  - before: identity（不旋转）量化模型
  - after : hadamard（旋转）量化模型

严格满足：
1) 每层每个量化投影层：按 hadamard_group_size 在最后一维（列/in_features）分块统计
2) 输出旋转前（before）与旋转后（after）的分布统计
3) 统计对象是 pseudoquant 的 dqweight（已映射到 mxfp4/nvfp4 上的反量化权重）

输出：
- CSV：每行对应 (layer, module, block_id) 的 before/after 统计 + ratio/delta
"""

import os
import json
import re
import csv
import math
import argparse
from typing import Dict, List, Tuple

import torch

DQ_SUFFIX = ".dqweight"
LAYER_KEY_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.dqweight$")

# ----------------------------
# IO helpers
# ----------------------------
def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_weight_map(model_dir: str) -> Dict[str, str]:
    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    idx = read_json(idx_path)
    return idx["weight_map"]

def load_quant_cfg(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "config.json")
    cfg = read_json(cfg_path)
    qc = cfg.get("quantization_config", {})
    return cfg, qc

def list_dqweight_keys(weight_map: Dict[str, str]) -> List[str]:
    return sorted([k for k in weight_map.keys() if k.endswith(DQ_SUFFIX)])

def parse_layer_and_module(dq_key: str) -> Tuple[int, str]:
    """
    model.layers.<i>.<module>.dqweight -> (i, <module>)
    """
    m = LAYER_KEY_RE.match(dq_key)
    if not m:
        raise ValueError(f"Unexpected dqweight key format: {dq_key}")
    return int(m.group(1)), m.group(2)

def load_tensor_safetensors(model_dir: str, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    """
    从 safetensors 分片中读取单个 tensor（不会加载整个 shard）。
    依赖 safetensors 已安装（你量化项目本身就依赖它，通常已存在）。
    """
    try:
        from safetensors import safe_open
    except Exception as e:
        raise RuntimeError(
            "找不到 safetensors。请在你的环境中安装：pip install safetensors\n"
            f"原始错误：{e}"
        )

    shard = weight_map.get(key, None)
    if shard is None:
        raise KeyError(f"Key not found in weight_map: {key}")

    shard_path = os.path.join(model_dir, shard)
    # safe_open 会只读取所需 tensor
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)

# ----------------------------
# Stats helpers
# ----------------------------
@torch.no_grad()
def block_stats(x: torch.Tensor) -> Dict[str, float]:
    """
    x: 2D block slice, shape [out_features, block_size]
    只做轻量统计，保证整体运行时间不长：
      - mean/std/min/max
      - abs_mean/abs_max
      - zero_frac
      - frac(|x|>thr) for a few thresholds
    """
    # 用 float32 统计，避免 bf16/fp16 数值误差
    xf = x.float()
    absx = xf.abs()

    n = xf.numel()
    if n == 0:
        return {}

    # 基础统计
    mean = xf.mean().item()
    std = xf.std(unbiased=False).item()
    mn = xf.min().item()
    mx = xf.max().item()

    abs_mean = absx.mean().item()
    abs_max = absx.max().item()

    zero_frac = (xf == 0).float().mean().item()

    # 尾部比例（阈值可按你的需要再加/改）
    thrs = [0.05, 0.1,0.2, 0.4, 0.5]
    tail = {f"frac_abs_gt_{t:g}": (absx > t).float().mean().item() for t in thrs}

    return {
        "n": float(n),
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "abs_mean": abs_mean,
        "abs_max": abs_max,
        "zero_frac": zero_frac,
        **tail,
    }

def safe_ratio(a: float, b: float) -> float:
    if b == 0.0:
        return math.nan
    return a / b

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before_model_dir", type=str, required=True,
                    help="旋转前（identity）pseudoquant 模型目录（包含 dqweight）")
    ap.add_argument("--after_model_dir", type=str, required=True,
                    help="旋转后（hadamard）pseudoquant 模型目录（包含 dqweight）")
    ap.add_argument("--out_csv", type=str, default="fp4_weight_block_stats_before_after.csv",
                    help="输出 CSV 路径")
    ap.add_argument("--only_modules", type=str, default="",
                    help="可选：正则过滤 module 名（例如 'self_attn\\.(q|k|v)_proj'）")
    ap.add_argument("--limit_layers", type=int, default=-1,
                    help="可选：只统计前 N 层（调试用）；默认 -1 表示所有层")
    ap.add_argument("--torch_threads", type=int, default=0,
                    help="可选：设置 torch CPU 线程数（0 表示不改）")
    args = ap.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    # 读 config
    before_cfg, before_qc = load_quant_cfg(args.before_model_dir)
    after_cfg,  after_qc  = load_quant_cfg(args.after_model_dir)

    # 基本一致性检查（避免你统计错模型）
    if not before_qc.get("pseudoquantization", False) or not after_qc.get("pseudoquantization", False):
        raise RuntimeError("before/after 必须都是 pseudoquantization=True 的导出模型（需要 dqweight）。")

    h = int(after_qc.get("hadamard_group_size", 0))
    if h <= 0:
        raise RuntimeError("after 模型 config 里未找到 hadamard_group_size。")

    # 读 weight_map
    before_wm = load_weight_map(args.before_model_dir)
    after_wm  = load_weight_map(args.after_model_dir)

    before_keys = set(list_dqweight_keys(before_wm))
    after_keys  = set(list_dqweight_keys(after_wm))

    common_keys = sorted(list(before_keys.intersection(after_keys)))
    if not common_keys:
        raise RuntimeError("before/after 没有共同的 dqweight keys。请确认两个目录都是 pseudoquant 导出，并且模型结构相同。")

    # 可选 module 过滤
    mod_re = re.compile(args.only_modules) if args.only_modules else None

    # 输出 CSV
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # 表头：基本信息 + before stats + after stats + ratio/delta（挑关键项）
        base_cols = ["layer", "module", "block_id", "col_start", "col_end", "out_features", "in_features", "block_size"]
        stat_cols = [
            "n", "mean", "std", "min", "max", "abs_mean", "abs_max", "zero_frac",
            "frac_abs_gt_0.05", "frac_abs_gt_0.1", "frac_abs_gt_0.2", "frac_abs_gt_0.4", "frac_abs_gt_0.5"
        ]
        writer.writerow(
            base_cols
            + [f"before_{c}" for c in stat_cols]
            + [f"after_{c}" for c in stat_cols]
            + [
                # 你后续选“哪些块需要旋转”时最常用的对比指标
                "ratio_abs_max(after/before)",
                "ratio_std(after/before)",
                "delta_zero_frac(after-before)",
                "ratio_frac_abs_gt_0.4(after/before)",
                "ratio_frac_abs_gt_0.5(after/before)",
            ]
        )

        # 主循环：逐 tensor、逐块统计（不做 Hadamard 变换，速度快很多）
        for key in common_keys:
            layer, module = parse_layer_and_module(key)
            if args.limit_layers > 0 and layer >= args.limit_layers:
                continue
            if mod_re and not mod_re.search(module):
                continue

            # 加载 before/after dqweight（已是 FP4 映射后的反量化权重）
            w_before = load_tensor_safetensors(args.before_model_dir, before_wm, key)
            w_after  = load_tensor_safetensors(args.after_model_dir,  after_wm,  key)

            if w_before.shape != w_after.shape:
                raise RuntimeError(f"Shape mismatch for {key}: before {tuple(w_before.shape)} vs after {tuple(w_after.shape)}")

            # 期望形状为 [out_features, in_features]
            if w_before.ndim != 2:
                raise RuntimeError(f"Expect 2D weight for {key}, got shape {tuple(w_before.shape)}")

            out_features, in_features = w_before.shape
            if in_features % h != 0:
                raise RuntimeError(
                    f"in_features ({in_features}) not divisible by hadamard_group_size ({h}) for {key}."
                )

            n_blocks = in_features // h

            # 逐 block 统计
            for bid in range(n_blocks):
                s = bid * h
                e = (bid + 1) * h

                b0 = w_before[:, s:e]
                b1 = w_after[:,  s:e]

                st0 = block_stats(b0)
                st1 = block_stats(b1)

                # 派生对比指标（用于后续筛块）
                ratio_abs_max = safe_ratio(st1["abs_max"], st0["abs_max"])
                ratio_std = safe_ratio(st1["std"], st0["std"])
                delta_zero = st1["zero_frac"] - st0["zero_frac"]
                ratio_tail0_4 = safe_ratio(st1["frac_abs_gt_0.4"], st0["frac_abs_gt_0.5"])
                ratio_tail0_5 = safe_ratio(st1["frac_abs_gt_0.5"], st0["frac_abs_gt_0.5"])

                row = [
                    layer, module, bid, s, e, out_features, in_features, h,
                    # before stats
                    st0["n"], st0["mean"], st0["std"], st0["min"], st0["max"], st0["abs_mean"], st0["abs_max"], st0["zero_frac"],
                    st0["frac_abs_gt_0.05"], st0["frac_abs_gt_0.1"], st0["frac_abs_gt_0.2"], st0["frac_abs_gt_0.4"], st0["frac_abs_gt_0.5"],
                    # after stats
                    st1["n"], st1["mean"], st1["std"], st1["min"], st1["max"], st1["abs_mean"], st1["abs_max"], st1["zero_frac"],
                    st1["frac_abs_gt_0.05"], st1["frac_abs_gt_0.1"], st1["frac_abs_gt_0.2"], st1["frac_abs_gt_0.4"], st1["frac_abs_gt_0.5"],
                    # compare
                    ratio_abs_max, ratio_std, delta_zero, ratio_tail0_4, ratio_tail0_5
                ]
                writer.writerow(row)

    print(f"[OK] Wrote per-block before/after FP4-mapped weight stats to: {args.out_csv}")
    print(f"[INFO] hadamard_group_size (column block size) = {h}")
    print("[INFO] Interpretation tip: if a block has ratio_abs_max<<1 and/or tail ratios <<1, rotation likely reduced outliers in that block.")

if __name__ == "__main__":
    main()

