#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot per-output-channel outliers for WHOLE dqweight tensor (no hadamard blocks).

For each selected (layer, module) dqweight:
  x-axis: output channel index (row index) normalized to [0,1]
  y-axis: row_abs_max[i] = max_j |W[i, j]|  (channel outlier)

We plot BEFORE vs AFTER in two separate subplots (colors differ),
and use y-limit = global_max * y_pad to avoid truncation.

Selection (sampling):
  - Filter by --modules_re and --layers
  - Optionally sample layers: --sample_layers
  - Optionally sample modules per layer: --rand_per_layer
  - Optionally cap total pairs: --max_pairs

Output:
  Multi-page PDF: <out_dir>/<pdf_name>

Example:
python plot_layer_channel_outliers.py \
  --before_dir /path/to/identity_model \
  --after_dir  /path/to/hadamard_model \
  --out_dir plots_layer \
  --modules_re "mlp\\.(down_proj|gate_proj|up_proj)|self_attn\\.(q_proj|k_proj|v_proj|o_proj)" \
  --sample_layers 8 \
  --rand_per_layer 2 \
  --seed 123
"""

import os
import re
import math
import json
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from safetensors import safe_open

DQ_SUFFIX = ".dqweight"
KEY_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)\.dqweight$")


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


def parse_layer_module_from_key(key: str) -> Optional[Tuple[int, str]]:
    m = KEY_RE.match(key)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def parse_layers_spec(layers: str) -> Optional[set]:
    if not layers:
        return None
    layers = layers.strip()
    if "-" in layers:
        a, b = layers.split("-")
        return set(range(int(a), int(b) + 1))
    return set(int(x) for x in layers.split(",") if x.strip())


# ----------------------------
# Selection helpers
# ----------------------------
def collect_common_dqkeys(wm_b: Dict[str, str], wm_a: Dict[str, str]) -> List[str]:
    kb = {k for k in wm_b.keys() if k.endswith(DQ_SUFFIX)}
    ka = {k for k in wm_a.keys() if k.endswith(DQ_SUFFIX)}
    common = sorted(kb.intersection(ka))
    if not common:
        raise RuntimeError("No common dqweight keys between before/after.")
    return common


def select_layer_module_pairs(
    keys: List[str],
    modules_re: str,
    layers: str,
    sample_layers: int,
    rand_per_layer: int,
    max_pairs: int,
    seed: int,
) -> List[Tuple[int, str, str]]:
    """
    Returns list of (layer, module, full_key).
    Sampling order:
      1) filter by modules_re and layers
      2) sample layers if requested
      3) per chosen layer, sample modules if rand_per_layer>0
      4) cap total pairs if max_pairs>0
    """
    mod_pat = re.compile(modules_re) if modules_re else None
    layer_set = parse_layers_spec(layers)

    # parse & filter
    candidates = []
    for k in keys:
        lm = parse_layer_module_from_key(k)
        if lm is None:
            continue
        layer, module = lm
        if layer_set is not None and layer not in layer_set:
            continue
        if mod_pat is not None and not mod_pat.search(module):
            continue
        candidates.append((layer, module, k))

    if not candidates:
        raise RuntimeError("No (layer,module) left after filtering (--modules_re/--layers).")

    # group by layer
    by_layer = defaultdict(list)
    for layer, module, k in candidates:
        by_layer[layer].append((layer, module, k))

    all_layers = sorted(by_layer.keys())
    rng = random.Random(seed)

    # sample layers
    if sample_layers and sample_layers > 0 and sample_layers < len(all_layers):
        rng.shuffle(all_layers)
        all_layers = sorted(all_layers[:sample_layers])

    chosen = []
    for layer in all_layers:
        items = by_layer[layer]
        if rand_per_layer and rand_per_layer > 0 and rand_per_layer < len(items):
            rng.shuffle(items)
            items = items[:rand_per_layer]
        chosen.extend(items)

    # cap total pairs
    if max_pairs and max_pairs > 0 and max_pairs < len(chosen):
        rng.shuffle(chosen)
        chosen = chosen[:max_pairs]

    # stable sort for nicer PDF order
    chosen.sort(key=lambda x: (x[0], x[1]))
    return chosen


# ----------------------------
# Plot helpers
# ----------------------------
def downsample_curve(y: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    n = y.size
    if n == 0:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32)
    if max_points <= 0 or n <= max_points:
        x = np.linspace(0.0, 1.0, n, endpoint=True) if n > 1 else np.array([0.0])
        return x.astype(np.float32), y.astype(np.float32)

    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    x = idx / (n - 1) if n > 1 else idx.astype(np.float32)
    return x.astype(np.float32), y[idx].astype(np.float32)


def set_ylim_maxpad(ax, ymax: float, pad: float, ymin: float = 0.0):
    if not math.isfinite(ymax) or ymax <= 0:
        return
    ax.set_ylim(bottom=ymin, top=ymax * pad)


def plot_one_page(
    pdf: PdfPages,
    title: str,
    rowmax_b: np.ndarray,
    rowmax_a: np.ndarray,
    max_points: int,
    y_pad: float,
    annotate: Dict[str, str],
):
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.28)

    axb = fig.add_subplot(gs[0, 0])
    axa = fig.add_subplot(gs[1, 0])

    xb, yb = downsample_curve(rowmax_b, max_points)
    xa, ya = downsample_curve(rowmax_a, max_points)

    axb.plot(xb, yb, linewidth=0.7, label="before(identity)",color='orange')
    axa.plot(xa, ya, linewidth=0.7, label="after(hadamard)")

    axb.set_ylabel("row abs_max  (max_j |W[i,j]|)")
    axa.set_ylabel("row abs_max  (max_j |W[i,j]|)")
    axa.set_xlabel("output channel index (normalized)")

    axb.grid(True, linewidth=0.3, alpha=0.4)
    axa.grid(True, linewidth=0.3, alpha=0.4)

    axb.legend(loc="upper right", frameon=True)
    axa.legend(loc="upper right", frameon=True)

    # avoid truncation: use true global max * pad
    y_global_max = float(max(np.max(rowmax_b), np.max(rowmax_a)))
    set_ylim_maxpad(axb, y_global_max, y_pad, ymin=0.0)
    set_ylim_maxpad(axa, y_global_max, y_pad, ymin=0.0)

    fig.suptitle(title, fontsize=11)

    if annotate:
        txt = "\n".join([f"{k}: {v}" for k, v in list(annotate.items())[:14]])
        axb.text(
            0.99, 0.02, txt,
            transform=axb.transAxes,
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

    ap.add_argument("--before_dir", required=True, help="identity pseudoquant 模型目录")
    ap.add_argument("--after_dir", required=True, help="hadamard pseudoquant 模型目录")
    ap.add_argument("--out_dir", default="plots_layer", help="输出目录")
    ap.add_argument("--pdf_name", default="layer_channel_outliers.pdf", help="输出 PDF 文件名")

    # sampling / filtering
    ap.add_argument("--modules_re", default="", help="可选：模块过滤正则")
    ap.add_argument("--layers", default="", help="可选：层过滤，如 '0-7' 或 '0,1,2'")
    ap.add_argument("--sample_layers", type=int, default=0, help="可选：随机抽样 N 层（0=不抽样）")
    ap.add_argument("--rand_per_layer", type=int, default=0, help="可选：每层随机抽 K 个 module（0=不抽样）")
    ap.add_argument("--max_pairs", type=int, default=0, help="可选：全局最多画多少个(layer,module)（0=不限制）")

    # plot controls
    ap.add_argument("--max_points", type=int, default=4096, help="row_max 曲线最多画多少点（仅绘图降采样）")
    ap.add_argument("--y_pad", type=float, default=1.15, help="y上界 = global_max * y_pad，调大可避免截断")

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

    wm_b = read_weight_map(args.before_dir)
    wm_a = read_weight_map(args.after_dir)

    common_keys = collect_common_dqkeys(wm_b, wm_a)

    chosen = select_layer_module_pairs(
        keys=common_keys,
        modules_re=args.modules_re,
        layers=args.layers,
        sample_layers=args.sample_layers,
        rand_per_layer=args.rand_per_layer,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )

    out_pdf = os.path.join(args.out_dir, args.pdf_name)
    num_pages = 0
    skipped = 0

    with PdfPages(out_pdf) as pdf:
        for layer, module, key in chosen:
            if key not in wm_b or key not in wm_a:
                skipped += 1
                continue

            Wb = load_tensor(args.before_dir, wm_b, key)
            Wa = load_tensor(args.after_dir,  wm_a, key)

            if Wb.ndim != 2 or Wa.ndim != 2 or Wb.shape != Wa.shape:
                skipped += 1
                continue

            out_f, in_f = Wb.shape

            # row abs_max (channel outlier)
            # use float32 reduction for stability
            rowmax_b = Wb.abs().to(torch.float32).amax(dim=1).cpu().numpy()
            rowmax_a = Wa.abs().to(torch.float32).amax(dim=1).cpu().numpy()

            title = f"layer {layer:02d} | {module} | row_outliers (abs_max per output channel)"

            annotate = {
                "tensor": key,
                "shape": f"{out_f}x{in_f}",
                "before_abs_max": f"{float(np.max(rowmax_b)):.6g}",
                "after_abs_max":  f"{float(np.max(rowmax_a)):.6g}",
                "before_zero_frac": f"{float((Wb == 0).to(torch.float32).mean().item()):.6g}",
                "after_zero_frac":  f"{float((Wa == 0).to(torch.float32).mean().item()):.6g}",
                "max_points": str(args.max_points),
                "y_pad": str(args.y_pad),
            }

            plot_one_page(
                pdf=pdf,
                title=title,
                rowmax_b=rowmax_b,
                rowmax_a=rowmax_a,
                max_points=args.max_points,
                y_pad=args.y_pad,
                annotate=annotate,
            )
            num_pages += 1

    print(f"[OK] wrote {num_pages} pages to: {out_pdf}")
    print(f"[INFO] forward_dtype={qc_a.get('forward_dtype')}, seed={args.seed}")
    if args.sample_layers:
        print(f"[INFO] sampled layers: {args.sample_layers}")
    if args.rand_per_layer:
        print(f"[INFO] random modules per layer: {args.rand_per_layer}")
    if args.max_pairs:
        print(f"[INFO] capped pairs: {args.max_pairs}")
    if skipped:
        print(f"[WARN] skipped: {skipped} (missing keys or shape mismatch)")


if __name__ == "__main__":
    main()
