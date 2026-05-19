"""
Usage example:

python plot_section_3_3_quant_error.py ^
  --input-dir ./section_3_3_quant_error_outputs/mxfp_g32 ^
  --output-dir ./generated_figures/section_3_3_mxfp_g32

python plot_section_3_3_quant_error.py ^
  --input-dir ./section_3_3_quant_error_outputs/nvfp_g16 ^
  --output-dir ./generated_figures/section_3_3_nvfp_g16
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


MODEL_ORDER = ["llama-2-7b-hf", "llama-3-8b-hf", "Qwen3-8B"]
MODEL_DISPLAY = {
    "llama-2-7b-hf": "Llama-2-7B-HF",
    "llama-3-8b-hf": "Llama-3-8B-HF",
    "Qwen3-8B": "Qwen3-8B",
}
SLOT_DISPLAY = {
    "qkv": "qkv input",
    "o": "o_proj input",
    "gate_up": "gate/up input",
    "down": "down_proj input",
}
PATTERN_DISPLAY = {
    "outlier_squeezing": "Outlier squeezing",
    "large_value_dominant": "Large-value dominant",
    "mixed": "Mixed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot group-wise activation quantization-error figures for thesis Section 3.3."
    )
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--slots", nargs="+", default=None, choices=["qkv", "o", "gate_up", "down"])
    parser.add_argument("--layer-indices", nargs="+", type=int, default=None)
    parser.add_argument("--figure-dpi", type=int, default=220)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def to_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def sanitize_case(model: str, layer_idx: int, slot: str, fmt: str, group_size: int) -> str:
    model_key = model.replace("-", "_").replace("/", "_")
    return f"{model_key}_b{layer_idx:02d}_{slot}_{fmt}_g{group_size}"


def case_sort_key(case_key: tuple[str, int, str]) -> tuple[int, int, str]:
    model, layer_idx, slot = case_key
    try:
        model_rank = MODEL_ORDER.index(model)
    except ValueError:
        model_rank = len(MODEL_ORDER)
    return (model_rank, layer_idx, slot)


def filter_rows(
    rows: list[dict[str, str]],
    models: list[str] | None,
    slots: list[str] | None,
    layer_indices: list[int] | None,
) -> list[dict[str, str]]:
    out = []
    for row in rows:
        model = row["model"]
        slot = row["slot"]
        layer_idx = to_int(row, "layer_idx")
        if models and model not in models:
            continue
        if slots and slot not in slots:
            continue
        if layer_indices and layer_idx not in layer_indices:
            continue
        out.append(row)
    return out


def plot_group_error_vs_max(
    case_rows: list[dict[str, str]],
    case_summary: dict[str, str],
    output_dir: Path,
    dpi: int,
) -> None:
    case_rows = sorted(case_rows, key=lambda row: to_int(row, "group_idx"))
    model = case_rows[0]["model"]
    layer_idx = to_int(case_rows[0], "layer_idx")
    slot = case_rows[0]["slot"]
    fmt = case_rows[0]["format"]
    group_size = to_int(case_rows[0], "group_size")

    x = [to_int(row, "group_idx") for row in case_rows]
    mse = [to_float(row, "avg_group_mse") for row in case_rows]
    max_abs = [to_float(row, "avg_group_max_abs") for row in case_rows]
    corr = to_float(case_summary, "pearson_group_mse_vs_max_abs")

    fig, ax1 = plt.subplots(figsize=(10.5, 4.6))
    ax2 = ax1.twinx()

    ax1.bar(
        x,
        mse,
        width=0.82,
        color="#c44e52",
        edgecolor="#8f2d32",
        linewidth=0.6,
        alpha=0.85,
        label="Quantization error",
    )
    ax2.plot(
        x,
        max_abs,
        color="#4c72b0",
        linewidth=1.8,
        marker="o",
        markersize=3.0,
        label="Max value",
    )

    ax1.set_xlabel("Group index")
    ax1.set_ylabel("Average group quantization MSE", color="#8f2d32")
    ax2.set_ylabel("Average group max-abs value", color="#355c96")
    ax1.tick_params(axis="y", labelcolor="#8f2d32")
    ax2.tick_params(axis="y", labelcolor="#355c96")
    ax1.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    title = (
        f"{MODEL_DISPLAY.get(model, model)} | block {layer_idx} | "
        f"{SLOT_DISPLAY.get(slot, slot)}"
    )
    subtitle = f"{fmt.upper()}, group size = {group_size}, corr(error, max) = {corr:.3f}"
    fig.suptitle(title, fontsize=16, y=0.98)
    ax1.set_title(subtitle, fontsize=11, pad=10)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)

    fig.tight_layout()
    stem = sanitize_case(model, layer_idx, slot, fmt, group_size)
    fig.savefig(output_dir / f"fig3_4_qerror_vs_max_{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"fig3_4_qerror_vs_max_{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_top_group_zoom_in(
    bin_rows: list[dict[str, str]],
    top_rows: list[dict[str, str]],
    output_dir: Path,
    dpi: int,
) -> None:
    model = bin_rows[0]["model"]
    layer_idx = to_int(bin_rows[0], "layer_idx")
    slot = bin_rows[0]["slot"]
    fmt = bin_rows[0]["format"]
    group_size = to_int(bin_rows[0], "group_size")

    top_rows = sorted(top_rows, key=lambda row: to_int(row, "top_rank"))
    panels = len(top_rows)
    ncols = min(3, panels)
    nrows = math.ceil(panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)

    grouped_bins = defaultdict(list)
    for row in bin_rows:
        grouped_bins[to_int(row, "group_idx")].append(row)

    for ax in axes.flat:
        ax.axis("off")

    for ax, top_row in zip(axes.flat, top_rows):
        ax.axis("on")
        group_idx = to_int(top_row, "group_idx")
        rows = sorted(grouped_bins[group_idx], key=lambda row: to_int(row, "bin_idx"))
        bin_centers = [(to_float(row, "bin_left") + to_float(row, "bin_right")) / 2.0 for row in rows]
        value_ratio = [to_float(row, "value_count_ratio") for row in rows]
        error_ratio = [to_float(row, "error_contribution_ratio") for row in rows]

        ax.axvspan(0.0, 0.3, color="#d9e6f5", alpha=0.35, lw=0)
        ax.axvspan(0.7, 1.0, color="#f7e1d5", alpha=0.35, lw=0)
        ax.bar(
            bin_centers,
            value_ratio,
            width=0.085,
            color="#b9bec7",
            edgecolor="#6f7782",
            linewidth=0.6,
            alpha=0.9,
            label="Value ratio",
        )
        ax.plot(
            bin_centers,
            error_ratio,
            color="#c44e52",
            linewidth=1.8,
            marker="o",
            markersize=3.2,
            label="Error contribution ratio",
        )

        pattern_code = top_row["pattern_code"]
        pattern_name = PATTERN_DISPLAY.get(pattern_code, pattern_code)
        top_rank = to_int(top_row, "top_rank")
        avg_mse = to_float(top_row, "avg_group_mse")
        avg_max = to_float(top_row, "avg_group_max_abs")
        ax.set_title(
            f"Top {top_rank} | G{group_idx} | {pattern_name}\n"
            f"MSE={avg_mse:.4e}, Max={avg_max:.3f}",
            fontsize=10,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, max(max(value_ratio), max(error_ratio)) * 1.18 if rows else 1.0)
        ax.set_xlabel("Normalized magnitude")
        ax.set_ylabel("Ratio")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([0.05, 0.25, 0.45, 0.65, 0.85])
        ax.set_xticklabels(["0.0-0.1", "0.2-0.3", "0.4-0.5", "0.6-0.7", "0.8-0.9"], fontsize=8)

    title = (
        f"{MODEL_DISPLAY.get(model, model)} | block {layer_idx} | "
        f"{SLOT_DISPLAY.get(slot, slot)}"
    )
    subtitle = f"Top-{len(top_rows)} high-error groups: normalized value distribution vs. error contribution"
    fig.suptitle(title, fontsize=16, y=0.995)
    fig.text(0.5, 0.955, subtitle, ha="center", fontsize=11)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=[0.02, 0.03, 1.0, 0.9])

    stem = sanitize_case(model, layer_idx, slot, fmt, group_size)
    fig.savefig(output_dir / f"fig3_5_zoom_in_{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_dir / f"fig3_5_zoom_in_{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_rows = read_csv_rows(input_dir / "group_error_stats.csv")
    case_rows = read_csv_rows(input_dir / "case_summary.csv")
    top_rows = read_csv_rows(input_dir / "top_group_summary.csv")
    bin_rows = read_csv_rows(input_dir / "top_group_bin_contrib.csv")

    group_rows = filter_rows(group_rows, args.models, args.slots, args.layer_indices)
    case_rows = filter_rows(case_rows, args.models, args.slots, args.layer_indices)
    top_rows = filter_rows(top_rows, args.models, args.slots, args.layer_indices)
    bin_rows = filter_rows(bin_rows, args.models, args.slots, args.layer_indices)

    case_group_rows = defaultdict(list)
    for row in group_rows:
        case_group_rows[(row["model"], to_int(row, "layer_idx"), row["slot"])].append(row)

    case_summary_map = {
        (row["model"], to_int(row, "layer_idx"), row["slot"]): row
        for row in case_rows
    }
    case_top_rows = defaultdict(list)
    for row in top_rows:
        case_top_rows[(row["model"], to_int(row, "layer_idx"), row["slot"])].append(row)

    case_bin_rows = defaultdict(list)
    for row in bin_rows:
        case_bin_rows[(row["model"], to_int(row, "layer_idx"), row["slot"])].append(row)

    case_keys = sorted(case_group_rows.keys(), key=case_sort_key)
    for case_key in case_keys:
        plot_group_error_vs_max(
            case_rows=case_group_rows[case_key],
            case_summary=case_summary_map[case_key],
            output_dir=output_dir,
            dpi=args.figure_dpi,
        )
        plot_top_group_zoom_in(
            bin_rows=case_bin_rows[case_key],
            top_rows=case_top_rows[case_key],
            output_dir=output_dir,
            dpi=args.figure_dpi,
        )

    print(f"[DONE] Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
