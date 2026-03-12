#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
utils:
python postprocess_block_csv.py \
  --in_csv fp4_block_stats_llama3_mxfp_gptq.csv \
  --out_dir . \
  --topk 5 \
  --sort_metric "ratio_abs_max(after/before)"

python postprocess_block_csv.py \
  --in_csv fp4_block_stats_llama3_mxfp_gptq.csv \
  --out_dir . \
  --topk 8 \
  --sort_metric "ratio_frac_abs_gt_0.4(after/before)"

Post-process the wide CSV produced by block stats script, and generate:
  1) compact.tsv : smaller, easier to read in vim/less
  2) summary.tsv : aggregated stats per (layer, module)
  3) report.txt  : top-K best/worst blocks per (layer, module)

Compatible with updated thresholds: 0.05/0.1/0.2/0.4/0.8
No heavy dependencies (stdlib only).
"""

import argparse
import csv
import math
from collections import defaultdict
from statistics import mean, median


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def quantile(values, p: float) -> float:
    """Simple quantile (p in [0,1]) for a list; ignores NaNs."""
    xs = sorted([v for v in values if not math.isnan(v)])
    if not xs:
        return float("nan")
    if p <= 0:
        return xs[0]
    if p >= 1:
        return xs[-1]
    k = (len(xs) - 1) * p
    i = int(math.floor(k))
    j = int(math.ceil(k))
    if i == j:
        return xs[i]
    return xs[i] * (j - k) + xs[j] * (k - i)


def fmt(x: float, nd=4) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.{nd}f}"


def sort_key_numeric(row: dict, metric: str, nan_to: float) -> float:
    """
    For stable sorting: map NaN -> nan_to (e.g., +inf so NaNs go to the end).
    """
    v = to_float(row.get(metric, "nan"))
    if math.isnan(v):
        return nan_to
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True, help="input wide CSV from stats script")
    ap.add_argument("--out_dir", default=".", help="output directory")
    ap.add_argument("--topk", type=int, default=5, help="top-K best/worst blocks per (layer,module)")
    ap.add_argument("--sort_metric", default="ratio_abs_max(after/before)",
                    help="metric to rank blocks")
    args = ap.parse_args()

    # === columns to keep in compact.tsv (updated) ===
    keep_cols = [
        "layer", "module", "block_id", "col_start", "col_end", "block_size",
        "before_abs_max", "after_abs_max",
        "before_std", "after_std",
        "before_zero_frac", "after_zero_frac",
        "ratio_abs_max(after/before)",
        "ratio_std(after/before)",
        "delta_zero_frac(after-before)",
        "ratio_frac_abs_gt_0.4(after/before)",
        "ratio_frac_abs_gt_0.5(after/before)",
        # 如果你希望 compact 里也看到更低阈值的 tail，可以加上：
        # "ratio_frac_abs_gt_0.05(after/before)",
        # "ratio_frac_abs_gt_0.1(after/before)",
        # "ratio_frac_abs_gt_0.2(after/before)",
    ]

    # === metrics to aggregate in summary.tsv (updated) ===
    agg_metrics = [
        "ratio_abs_max(after/before)",
        "ratio_frac_abs_gt_0.4(after/before)",
        "ratio_frac_abs_gt_0.5(after/before)",
        "delta_zero_frac(after-before)",
        "before_abs_max",
        "after_abs_max",
    ]

    # Read CSV
    rows = []
    with open(args.in_csv, "r", encoding="utf-8", newline="") as fcsv:
        reader = csv.DictReader(fcsv)
        header = reader.fieldnames or []
        required = set(keep_cols) | set(agg_metrics) | {
            "layer", "module", "block_id", "col_start", "col_end",
            "before_abs_max", "after_abs_max",
            "before_std", "after_std",
            "before_zero_frac", "after_zero_frac",
            "ratio_abs_max(after/before)",
            "ratio_frac_abs_gt_0.4(after/before)",
            "ratio_frac_abs_gt_0.5(after/before)",
            "delta_zero_frac(after-before)",
        }
        missing = [c for c in required if c not in header]
        if missing:
            raise RuntimeError(f"CSV missing required columns: {missing}")

        for r in reader:
            rows.append(r)

    # 1) Write compact.tsv
    out_dir = args.out_dir.rstrip("/")
    compact_path = f"{out_dir}/compact.tsv"
    with open(compact_path, "w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout, delimiter="\t")
        w.writerow(keep_cols)
        for r in rows:
            w.writerow([r[c] for c in keep_cols])

    # 2) Aggregate summary per (layer,module)
    group = defaultdict(list)
    for r in rows:
        group[(int(r["layer"]), r["module"])].append(r)

    summary_path = f"{out_dir}/summary.tsv"
    with open(summary_path, "w", encoding="utf-8", newline="") as fout:
        w = csv.writer(fout, delimiter="\t")
        w.writerow([
            "layer", "module", "num_blocks",
            *[f"{m}_mean" for m in agg_metrics],
            *[f"{m}_median" for m in agg_metrics],
            *[f"{m}_p10" for m in agg_metrics],
            *[f"{m}_p90" for m in agg_metrics],
        ])

        for (layer, module) in sorted(group.keys()):
            items = group[(layer, module)]
            wrow = [layer, module, len(items)]

            vals = {m: [to_float(it[m]) for it in items] for m in agg_metrics}

            # mean
            for m in agg_metrics:
                xs = [v for v in vals[m] if not math.isnan(v)]
                wrow.append(fmt(mean(xs)) if xs else "nan")
            # median
            for m in agg_metrics:
                xs = [v for v in vals[m] if not math.isnan(v)]
                wrow.append(fmt(median(xs)) if xs else "nan")
            # p10 / p90
            for m in agg_metrics:
                wrow.append(fmt(quantile(vals[m], 0.10)))
            for m in agg_metrics:
                wrow.append(fmt(quantile(vals[m], 0.90)))

            w.writerow(wrow)

    # 3) report.txt: Top-K best/worst blocks per (layer,module)
    metric = args.sort_metric
    if metric not in rows[0]:
        raise RuntimeError(f"sort_metric '{metric}' not found in CSV columns")

    report_path = f"{out_dir}/report.txt"
    with open(report_path, "w", encoding="utf-8") as fout:
        fout.write(f"Input: {args.in_csv}\n")
        fout.write(f"Ranking metric: {metric}\n")
        fout.write(f"TopK: {args.topk}\n\n")

        for (layer, module) in sorted(group.keys()):
            items = group[(layer, module)]

            # Stable sorting:
            # - "best": smallest metric, NaN goes to +inf (bottom)
            # - "worst": largest metric, NaN goes to -inf (bottom after reverse)
            items_sorted = sorted(items, key=lambda r: sort_key_numeric(r, metric, nan_to=float("inf")))

            fout.write(f"=== layer {layer:02d} | {module} ===\n")
            fout.write(f"Best blocks (smallest {metric}):\n")
            fout.write("  bid  cols        abs_max(b->a)     std(b->a)     zero(b->a)     r_abs_max   r_tail0.4   r_tail0.5   d_zero\n")

            for r in items_sorted[:args.topk]:
                bid = int(r["block_id"])
                cols = f'{r["col_start"]}-{r["col_end"]}'
                fout.write(
                    f"  {bid:>3d}  {cols:<10s}  "
                    f'{to_float(r["before_abs_max"]):>9.4f}->{to_float(r["after_abs_max"]):<9.4f}  '
                    f'{to_float(r["before_std"]):>8.4f}->{to_float(r["after_std"]):<8.4f}  '
                    f'{to_float(r["before_zero_frac"]):>7.4f}->{to_float(r["after_zero_frac"]):<7.4f}  '
                    f'{to_float(r["ratio_abs_max(after/before)"]):>8.4f}  '
                    f'{to_float(r["ratio_frac_abs_gt_0.4(after/before)"]):>9.4f}  '
                    f'{to_float(r["ratio_frac_abs_gt_0.5(after/before)"]):>9.4f}  '
                    f'{to_float(r["delta_zero_frac(after-before)"]):>7.4f}\n'
                )

            fout.write(f"\nWorst blocks (largest {metric}):\n")
            # for worst, sort with NaN -> -inf, then reverse to get largest first
            items_sorted_worst = sorted(items, key=lambda r: sort_key_numeric(r, metric, nan_to=float("-inf")))
            items_sorted_worst = list(reversed(items_sorted_worst))

            for r in items_sorted_worst[:args.topk]:
                bid = int(r["block_id"])
                cols = f'{r["col_start"]}-{r["col_end"]}'
                fout.write(
                    f"  {bid:>3d}  {cols:<10s}  "
                    f'{to_float(r["before_abs_max"]):>9.4f}->{to_float(r["after_abs_max"]):<9.4f}  '
                    f'{to_float(r["before_std"]):>8.4f}->{to_float(r["after_std"]):<8.4f}  '
                    f'{to_float(r["before_zero_frac"]):>7.4f}->{to_float(r["after_zero_frac"]):<7.4f}  '
                    f'{to_float(r["ratio_abs_max(after/before)"]):>8.4f}  '
                    f'{to_float(r["ratio_frac_abs_gt_0.4(after/before)"]):>9.4f}  '
                    f'{to_float(r["ratio_frac_abs_gt_0.5(after/before)"]):>9.4f}  '
                    f'{to_float(r["delta_zero_frac(after-before)"]):>7.4f}\n'
                )
            fout.write("\n\n")

    print(f"[OK] wrote:\n  {compact_path}\n  {summary_path}\n  {report_path}")
    print("[TIP] view compact.tsv via: less -S compact.tsv (or vim + :set nowrap)")
    print("[TIP] view summary.tsv via: column -t -s $'\\t' summary.tsv | less -S")


if __name__ == "__main__":
    main()

