#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot full-block dqweight distribution per Hadamard column-block, before vs after rotation.

Per selected block, one PDF page with 5 subplots:
  (1) row_max(|w|) vs output index (normalized)  [before only]
  (2) row_max(|w|) vs output index (normalized)  [after  only]
  (3) CCDF of |w| (log-y)                        [before vs after overlay]
  (4) |w| vs flattened index (normalized)        [before only]
  (5) |w| vs flattened index (normalized)        [after  only]

Key changes:
- Subplot1 split into two axes (before / after)
- Subplot3 uses |w| and flattened index, also split into two axes (before / after)
- y-limits for (1)(2)(4)(5) are set by TRUE max * pad, to avoid truncation
- optional downsampling ONLY for plotting curves (statistics still use full block)

This version (per your latest request):
- Fix subplot2 (CCDF) clipping: adds headroom/margins and robust x-limits
- Ensure subplot1 and subplot3 split-plots use different colors (before=C0, after=C1)

Usage example:
python plot_block_weight_distributions.py \
  --csv  fp4_block_stats_llama3_mxfp_gptq.csv \
  --before_dir /path/to/identity_model \
  --after_dir  /path/to/hadamard_model \
  --out_dir plots \
  --metric "ratio_abs_max(after/before)" \
  --mode best --topk 3 \
  --sample_layers 6 \
  --modules_re "mlp\\.(down_proj|gate_proj|up_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj)" \
  --seed 123 \
  --y_pad1 1.20 --y_pad3 1.20
"""

import os
import re
import csv
import math
import argparse
import random
import json
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from safetensors import safe_open

DQ_SUFFIX = ".dqweight"


# ----------------------------
# IO helpers
# ----------------------------
def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_quant_cfg(model_dir: str) -> dict:
    cfg = read_json(os.path.join(model_dir, "config.json"))
    return cfg.get("quantization_config", {})


def read_weight_map(model_dir: str) -> Dict[str, str]:
    idx = read_json(os.path.join(model_dir, "model.safetensors.index.json"))
    return idx["weight_map"]


def load_tensor(model_dir: str, weight_map: Dict[str, str], key: str) -> torch.Tensor:
    shard = weight_map.get(key)
    if shard is None:
        raise KeyError(f"Key not in weight_map: {key}")
    with safe_open(os.path.join(model_dir, shard), framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# ----------------------------
# CSV + selection
# ----------------------------
def parse_csv_rows(csv_path: str) -> List[dict]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [r for r in reader]


def parse_layers_spec(layers: str) -> Optional[set]:
    if not layers:
        return None
    layers = layers.strip()
    if "-" in layers:
        a, b = layers.split("-")
        return set(range(int(a), int(b) + 1))
    return set(int(x) for x in layers.split(",") if x.strip())


def stable_sort_key(v: float, nan_to: float) -> float:
    return nan_to if math.isnan(v) else v


def maybe_sample_layers(rows: List[dict], layers_set: Optional[set], sample_layers: int, seed: int) -> set:
    all_layers = sorted({int(r["layer"]) for r in rows})
    if layers_set is not None:
        all_layers = [l for l in all_layers if l in layers_set]
    if not all_layers:
        raise RuntimeError("No layers left after applying --layers filter.")

    if sample_layers and sample_layers > 0 and sample_layers < len(all_layers):
        rng = random.Random(seed)
        rng.shuffle(all_layers)
        return set(all_layers[:sample_layers])
    return set(all_layers)


def select_blocks(
    rows: List[dict],
    metric: str,
    modules_re: str,
    layers: str,
    sample_layers: int,
    mode: str,
    topk: int,
    rand_per_group: int,
    seed: int,
) -> List[dict]:
    mod_pat = re.compile(modules_re) if modules_re else None
    layers_set = parse_layers_spec(layers)

    filtered0 = []
    for r in rows:
        layer = int(r["layer"])
        module = r["module"]
        if layers_set is not None and layer not in layers_set:
            continue
        if mod_pat is not None and not mod_pat.search(module):
            continue
        filtered0.append(r)

    if not filtered0:
        raise RuntimeError("No rows left after filtering (check --modules_re / --layers).")

    final_layer_set = maybe_sample_layers(filtered0, layers_set, sample_layers, seed)
    filtered = [r for r in filtered0 if int(r["layer"]) in final_layer_set]
    if not filtered:
        raise RuntimeError("No rows left after layer sampling.")

    buckets = defaultdict(list)
    for r in filtered:
        buckets[(int(r["layer"]), r["module"])].append(r)

    rng = random.Random(seed)
    chosen = []

    if rand_per_group and rand_per_group > 0:
        for (_, _), items in sorted(buckets.items()):
            rng.shuffle(items)
            chosen.extend(items[:rand_per_group])
        if not chosen:
            raise RuntimeError("Selection is empty after rand_per_group sampling.")
        return chosen

    if mode not in ("best", "worst"):
        raise ValueError("mode must be 'best' or 'worst' when rand_per_group==0")

    for (_, _), items in sorted(buckets.items()):
        def keyfunc(it):
            return safe_float(it.get(metric, "nan"))

        if mode == "best":
            items_sorted = sorted(items, key=lambda it: stable_sort_key(keyfunc(it), nan_to=float("inf")))
            chosen.extend(items_sorted[:topk])
        else:
            items_sorted = sorted(items, key=lambda it: stable_sort_key(keyfunc(it), nan_to=float("-inf")))
            chosen.extend(items_sorted[-topk:])

    if not chosen:
        raise RuntimeError("Selection is empty (maybe metric missing or all NaN).")
    return chosen


# ----------------------------
# Curve helpers
# ----------------------------
def downsample_by_index(y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downsample y by evenly spaced indices.
    Return x in [0,1] normalized index, and y_sampled.
    """
    n = y.size
    if n == 0:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    if max_points <= 0 or n <= max_points:
        x = np.linspace(0.0, 1.0, n, endpoint=True) if n > 1 else np.array([0.0])
        return x.astype(np.float32), y.astype(np.float32)

    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    x = idx / (n - 1) if n > 1 else idx.astype(np.float32)
    return x.astype(np.float32), y[idx].astype(np.float32)


def ccdf_curve(abs_flat: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    CCDF: y = P(|w| > x). x is sorted |w| ascending.
    """
    s = np.sort(abs_flat.astype(np.float32, copy=False))
    n = s.size
    if n == 0:
        return np.array([0.0], dtype=np.float32), np.array([1.0], dtype=np.float32)

    if max_points <= 0 or n <= max_points:
        x = s
        y = 1.0 - (np.arange(n, dtype=np.float64) + 1.0) / float(n)
        y = np.maximum(y, 1e-12)
        return x.astype(np.float32), y.astype(np.float32)

    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    x = s[idx]
    y = 1.0 - (idx.astype(np.float64) + 1.0) / float(n)
    y = np.maximum(y, 1e-12)
    return x.astype(np.float32), y.astype(np.float32)


def set_ylim_maxpad(ax, y_max: float, pad: float, ymin: float = 0.0):
    if not math.isfinite(y_max) or y_max <= 0:
        return
    ax.set_ylim(bottom=ymin, top=y_max * pad)


# ----------------------------
# Plotting
# ----------------------------
def plot_block_page(
    pdf: PdfPages,
    title: str,
    row_max_before: np.ndarray,
    row_max_after: np.ndarray,
    abs_flat_before: np.ndarray,
    abs_flat_after: np.ndarray,
    max_curve_points: int,
    max_ccdf_points: int,
    max_flat_points: int,
    y_pad1: float,
    y_pad3: float,
    ccdf_x_qhi: float,
    ccdf_ymin: float,
    annotate: dict,
):
    """
    5 subplots per page.
    """
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0], hspace=0.35)

    ax1b = fig.add_subplot(gs[0, 0])  # row_max before
    ax1a = fig.add_subplot(gs[1, 0])  # row_max after
    ax2  = fig.add_subplot(gs[2, 0])  # ccdf overlay
    ax3b = fig.add_subplot(gs[3, 0])  # |w| vs flat idx before
    ax3a = fig.add_subplot(gs[4, 0])  # |w| vs flat idx after

    c_before, c_after = "C0", "C1"

    # ---------- (1) row_max before ----------
    x1b, y1b = downsample_by_index(row_max_before, max_curve_points)
    ax1b.plot(x1b, y1b, linewidth=0.7, color=c_before, label="before(identity)")
    ax1b.set_ylabel("max(|w|) per output channel")
    ax1b.set_xlabel("output channel index (normalized)")
    ax1b.grid(True, linewidth=0.3, alpha=0.4)
    ax1b.legend(loc="upper right", frameon=True)

    # ---------- (2) row_max after ----------
    x1a, y1a = downsample_by_index(row_max_after, max_curve_points)
    ax1a.plot(x1a, y1a, linewidth=0.7, color=c_after, label="after(hadamard)")
    ax1a.set_ylabel("max(|w|) per output channel")
    ax1a.set_xlabel("output channel index (normalized)")
    ax1a.grid(True, linewidth=0.3, alpha=0.4)
    ax1a.legend(loc="upper right", frameon=True)

    # y-limit for both row_max plots (avoid truncation)
    y1_global_max = float(max(np.max(row_max_before), np.max(row_max_after)))
    set_ylim_maxpad(ax1b, y1_global_max, pad=y_pad1, ymin=0.0)
    set_ylim_maxpad(ax1a, y1_global_max, pad=y_pad1, ymin=0.0)

    # ---------- (3) CCDF overlay (FIXED to avoid clipping) ----------
    x2b, y2b = ccdf_curve(abs_flat_before, max_ccdf_points)
    x2a, y2a = ccdf_curve(abs_flat_after,  max_ccdf_points)

    ax2.plot(x2b, y2b, linewidth=1.0, color=c_before, label="before(identity)")
    ax2.plot(x2a, y2a, linewidth=1.0, color=c_after,  label="after(hadamard)")
    ax2.set_yscale("log")
    ax2.set_ylabel("P(|w| > x)  (log scale)")
    ax2.set_xlabel("|w| threshold")
    ax2.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax2.legend(loc="upper right", frameon=True)

    # y-range: leave headroom above 1.0 to avoid top-edge clipping
    n_eff = max(int(abs_flat_before.size + abs_flat_after.size), 1)
    auto_ymin = max(1.0 / float(n_eff), 1e-12)
    ymin = auto_ymin if ccdf_ymin < 0 else max(ccdf_ymin, 1e-12)
    ax2.set_ylim(bottom=ymin, top=1.05)

    # x-range: quantile on positive values to avoid q==0 when many zeros
    def _pos_quantile(arr: np.ndarray, q: float) -> float:
        a = arr.astype(np.float64, copy=False)
        a = a[np.isfinite(a)]
        a = a[a > 0]
        if a.size == 0:
            return 0.0
        return float(np.quantile(a, q))

    qb = _pos_quantile(abs_flat_before, ccdf_x_qhi)
    qa = _pos_quantile(abs_flat_after,  ccdf_x_qhi)

    # ensure xlim also covers plotted curve ends
    x_end = 0.0
    if x2b.size:
        x_end = max(x_end, float(x2b[-1]))
    if x2a.size:
        x_end = max(x_end, float(x2a[-1]))

    xhi = max(qb, qa, x_end)
    if math.isfinite(xhi) and xhi > 0:
        ax2.set_xlim(left=0.0, right=xhi * 1.08)  # right padding avoids border clipping
        ax2.margins(x=0.01)
        if qb > 0:
            ax2.axvline(qb, color=c_before, linewidth=0.8, linestyle="--", alpha=0.7)
        if qa > 0:
            ax2.axvline(qa, color=c_after,  linewidth=0.8, linestyle="--", alpha=0.7)

    # ---------- (4) |w| vs flattened idx before ----------
    x3b, y3b = downsample_by_index(abs_flat_before, max_flat_points)
    ax3b.plot(x3b, y3b, linewidth=0.6, color=c_before, label="before(identity)")
    ax3b.set_ylabel("|w| (dqweight)")
    ax3b.set_xlabel("flattened index (normalized)")
    ax3b.grid(True, linewidth=0.3, alpha=0.4)
    ax3b.legend(loc="upper right", frameon=True)

    # ---------- (5) |w| vs flattened idx after ----------
    x3a, y3a = downsample_by_index(abs_flat_after, max_flat_points)
    ax3a.plot(x3a, y3a, linewidth=0.6, color=c_after, label="after(hadamard)")
    ax3a.set_ylabel("|w| (dqweight)")
    ax3a.set_xlabel("flattened index (normalized)")
    ax3a.grid(True, linewidth=0.3, alpha=0.4)
    ax3a.legend(loc="upper right", frameon=True)

    # y-limit for both flat plots (avoid truncation)
    y3_global_max = float(max(np.max(abs_flat_before), np.max(abs_flat_after)))
    set_ylim_maxpad(ax3b, y3_global_max, pad=y_pad3, ymin=0.0)
    set_ylim_maxpad(ax3a, y3_global_max, pad=y_pad3, ymin=0.0)

    fig.suptitle(title, fontsize=11)

    # annotate (place on first subplot)
    if annotate:
        lines = [f"{k}: {v}" for k, v in annotate.items()]
        txt = "\n".join(lines[:14])
        ax1b.text(
            0.99, 0.02, txt,
            transform=ax1b.transAxes,
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, linewidth=0.6)
        )

    pdf.savefig(fig)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True, help="block 统计 CSV（stat 脚本输出）")
    ap.add_argument("--before_dir", required=True, help="identity pseudoquant 模型目录")
    ap.add_argument("--after_dir", required=True, help="hadamard pseudoquant 模型目录")
    ap.add_argument("--out_dir", default="plots", help="输出目录")
    ap.add_argument("--pdf_name", default="block_dist_report.pdf", help="输出 PDF 文件名")

    # selection
    ap.add_argument("--metric", default="ratio_abs_max(after/before)", help="best/worst 排序指标（CSV 列名）")
    ap.add_argument("--modules_re", default="", help="可选：模块过滤正则")
    ap.add_argument("--layers", default="", help="可选：层过滤，如 '0-3' 或 '0,1,2'")
    ap.add_argument("--sample_layers", type=int, default=0, help="可选：随机抽样 N 层（0=不抽样）")

    ap.add_argument("--mode", choices=["best", "worst"], default="best",
                    help="当 --rand_per_group=0 时生效：每个(layer,module)取 best/worst TopK")
    ap.add_argument("--topk", type=int, default=3, help="best/worst: 每个(layer,module)选TopK块")
    ap.add_argument("--rand_per_group", type=int, default=0,
                    help=">0 则在每个(layer,module)随机抽 K 个块（优先于 mode/topk）")

    # plotting controls (downsample ONLY for drawing)
    ap.add_argument("--max_curve_points", type=int, default=4096,
                    help="row_max 曲线最多画多少点（仅绘图降采样）")
    ap.add_argument("--max_ccdf_points", type=int, default=4000,
                    help="CCDF 曲线最多画多少点（仅绘图降采样）")
    ap.add_argument("--max_flat_points", type=int, default=600000,
                    help="|w| vs flattened index 曲线最多画多少点（仅绘图降采样）")

    # y-limit pads to avoid truncation (NOTE: these are multiplicative pads)
    ap.add_argument("--y_pad1", type=float, default=1.20,
                    help="子图(1)(2) y上界 = max(row_max)*y_pad1，调大可避免截断")
    ap.add_argument("--y_pad3", type=float, default=1.20,
                    help="子图(4)(5) y上界 = max(|w|)*y_pad3，调大可避免截断")

    # CCDF x-limit quantile (readability)
    ap.add_argument("--ccdf_x_qhi", type=float, default=0.995,
                    help="CCDF 图 x 轴上界取 |w| 的该分位数（更可读），并画虚线标注")
    ap.add_argument("--ccdf_ymin", type=float, default=-1.0,
                    help="CCDF y轴下界；<0 自动用 max(1/N,1e-12)")

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--torch_threads", type=int, default=0)
    args = ap.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    os.makedirs(args.out_dir, exist_ok=True)

    # sanity check configs
    qc_b = read_quant_cfg(args.before_dir)
    qc_a = read_quant_cfg(args.after_dir)
    if not qc_b.get("pseudoquantization", False) or not qc_a.get("pseudoquantization", False):
        raise RuntimeError("before/after 都必须是 pseudoquantization=True (dqweight).")
    if qc_b.get("forward_dtype") != qc_a.get("forward_dtype"):
        raise RuntimeError(f"forward_dtype mismatch: {qc_b.get('forward_dtype')} vs {qc_a.get('forward_dtype')}")
    h = int(qc_a.get("hadamard_group_size", 0))
    if h <= 0:
        raise RuntimeError("after config missing hadamard_group_size.")

    wm_b = read_weight_map(args.before_dir)
    wm_a = read_weight_map(args.after_dir)

    rows = parse_csv_rows(args.csv)
    chosen = select_blocks(
        rows=rows,
        metric=args.metric,
        modules_re=args.modules_re,
        layers=args.layers,
        sample_layers=args.sample_layers,
        mode=args.mode,
        topk=args.topk,
        rand_per_group=args.rand_per_group,
        seed=args.seed,
    )

    # group chosen blocks by dqweight key so we load each tensor once
    by_key = defaultdict(list)
    for r in chosen:
        layer = int(r["layer"])
        module = r["module"]
        key = f"model.layers.{layer}.{module}{DQ_SUFFIX}"
        by_key[key].append(r)

    out_pdf = os.path.join(args.out_dir, args.pdf_name)
    num_pages = 0
    skipped = 0

    with PdfPages(out_pdf) as pdf:
        for key, block_rows in sorted(by_key.items()):
            if key not in wm_b or key not in wm_a:
                skipped += len(block_rows)
                continue

            Wb = load_tensor(args.before_dir, wm_b, key)
            Wa = load_tensor(args.after_dir,  wm_a, key)

            if Wb.ndim != 2 or Wa.ndim != 2 or Wb.shape != Wa.shape:
                skipped += len(block_rows)
                continue

            out_f, in_f = Wb.shape

            for r in block_rows:
                layer = int(r["layer"])
                module = r["module"]
                bid = int(r["block_id"])
                col_s = int(r["col_start"])
                col_e = int(r["col_end"])

                if col_e - col_s != h:
                    skipped += 1
                    continue

                # FULL block (no within-block sampling for stats)
                Bb = Wb[:, col_s:col_e].float().cpu().numpy()
                Ba = Wa[:, col_s:col_e].float().cpu().numpy()

                # row_max(|w|)
                row_max_b = np.max(np.abs(Bb), axis=1)
                row_max_a = np.max(np.abs(Ba), axis=1)

                # flat |w|
                abs_flat_b = np.abs(Bb.reshape(-1))
                abs_flat_a = np.abs(Ba.reshape(-1))

                mval = r.get(args.metric, "nan")
                title = (f"layer {layer:02d} | {module} | block {bid} cols[{col_s},{col_e}) "
                         f"| {args.metric}={mval}")

                annotate = {
                    "shape": f"{out_f}x{in_f}",
                    "block": f"{out_f}x{h}  (cols {col_s}-{col_e})",
                    "before_abs_max": f"{float(np.max(abs_flat_b)):.6g}",
                    "after_abs_max":  f"{float(np.max(abs_flat_a)):.6g}",
                    "before_zero_frac": f"{float(np.mean(Bb == 0.0)):.6g}",
                    "after_zero_frac":  f"{float(np.mean(Ba == 0.0)):.6g}",
                    "flat_points": f"{abs_flat_b.size}",
                    "y_pad1/y_pad3": f"{args.y_pad1}/{args.y_pad3}",
                    "ccdf_x_qhi": f"{args.ccdf_x_qhi}",
                }

                plot_block_page(
                    pdf=pdf,
                    title=title,
                    row_max_before=row_max_b,
                    row_max_after=row_max_a,
                    abs_flat_before=abs_flat_b,
                    abs_flat_after=abs_flat_a,
                    max_curve_points=args.max_curve_points,
                    max_ccdf_points=args.max_ccdf_points,
                    max_flat_points=args.max_flat_points,
                    y_pad1=args.y_pad1,
                    y_pad3=args.y_pad3,
                    ccdf_x_qhi=args.ccdf_x_qhi,
                    ccdf_ymin=args.ccdf_ymin,
                    annotate=annotate,
                )
                num_pages += 1

    print(f"[OK] wrote {num_pages} pages to: {out_pdf}")
    print(f"[INFO] hadamard_group_size={h}, forward_dtype={qc_a.get('forward_dtype')}")
    if args.sample_layers:
        print(f"[INFO] layer sampling: {args.sample_layers} layers (seed={args.seed})")
    if args.rand_per_group and args.rand_per_group > 0:
        print(f"[INFO] random blocks per (layer,module): {args.rand_per_group}")
    else:
        print(f"[INFO] selection per (layer,module): mode={args.mode}, topk={args.topk}, metric={args.metric}")
    if skipped:
        print(f"[WARN] skipped blocks: {skipped} (missing keys or shape mismatch or bad block size)")


if __name__ == "__main__":
    main()
