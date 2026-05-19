from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
ACADEMIC_COLORS = ["#1f4e79", "#8a5a44", "#5b7f3a"]
ACADEMIC_HATCHES = ["", "//", ".."]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Section 3.2 activation-distribution figures from precomputed group statistics."
    )
    parser.add_argument("--input-dir", type=str, default=str(THIS_DIR / "section_3_2_outputs"))
    parser.add_argument("--output-dir", type=str, default=str(THIS_DIR / "generated_figures" / "section_3_2"))
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--slot", type=str, default="qkv", choices=["qkv", "o", "gate_up", "down"])
    parser.add_argument("--strong-outlier-multiplier", type=float, default=5.0)
    parser.add_argument("--case-layer-mode", type=str, default="largest_tail", choices=["middle", "largest_tail"])
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def configure_academic_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "grid.color": "#c7c7c7",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
        }
    )


def choose_case_layer(summary_df: pd.DataFrame, model: str, slot: str, mode: str) -> int:
    cur = summary_df[(summary_df["model"] == model) & (summary_df["slot"] == slot)]
    if cur.empty:
        raise ValueError(f"No rows found for model={model}, slot={slot}")
    layer_ids = sorted(cur["layer_idx"].unique().tolist())
    if mode == "middle":
        return int(layer_ids[len(layer_ids) // 2])
    cur = cur.sort_values("max_density_multiplier", ascending=False)
    return int(cur.iloc[0]["layer_idx"])


def render_summary_figure(raw_df: pd.DataFrame, summary_df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    raw_df = raw_df[raw_df["group_size"] == args.group_size].copy()
    summary_df = summary_df[summary_df["group_size"] == args.group_size].copy()
    models = sorted(raw_df["model"].unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.9))
    ax1, ax2, ax3, ax4 = axes.ravel()

    slot_summary = summary_df.groupby(["model", "layer_idx"], as_index=False).agg(
        fraction_groups_density_gt_strong=("fraction_groups_density_gt_strong", "mean"),
        fraction_groups_density_le_1=("fraction_groups_density_le_1", "mean"),
    )
    block_ids = sorted(slot_summary["layer_idx"].unique().tolist())
    x = np.arange(len(block_ids))
    width = 0.24

    for idx, model in enumerate(models):
        cur = (
            slot_summary[slot_summary["model"] == model]
            .set_index("layer_idx")
            .reindex(block_ids)
        )
        ax1.bar(
            x + idx * width - width,
            cur["fraction_groups_density_gt_strong"].to_numpy(),
            width=width,
            color=ACADEMIC_COLORS[idx % len(ACADEMIC_COLORS)],
            edgecolor="#2f2f2f",
            linewidth=0.6,
            hatch=ACADEMIC_HATCHES[idx % len(ACADEMIC_HATCHES)],
            label=model,
        )
    ax1.set_title("(a) Fraction of strong outlier-concentrated groups across sampled transformer blocks")
    ax1.set_xlabel("Transformer block index")
    ax1.set_ylabel(f"Fraction (density multiplier > {args.strong_outlier_multiplier:g})")
    ax1.set_xticks(x, [str(v) for v in block_ids])
    ax1.grid(axis="y", alpha=0.35)
    ax1.legend(fontsize=9)

    for idx, model in enumerate(models):
        cur = raw_df[raw_df["model"] == model]["outlier_density_multiplier"].to_numpy()
        cur = np.sort(cur)
        y = np.linspace(0.0, 1.0, len(cur), endpoint=True)
        ax2.plot(
            cur,
            y,
            linewidth=1.8,
            color=ACADEMIC_COLORS[idx % len(ACADEMIC_COLORS)],
            label=model,
        )
    ax2.axvline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax2.set_title("(b) CDF of group outlier-density multiplier")
    ax2.set_xlabel("Outlier-density multiplier")
    ax2.set_ylabel("CDF")
    ax2.grid(alpha=0.35)
    ax2.legend(fontsize=9)

    for idx, model in enumerate(models):
        cur = (
            slot_summary[slot_summary["model"] == model]
            .set_index("layer_idx")
            .reindex(block_ids)
        )
        ax3.bar(
            x + idx * width - width,
            cur["fraction_groups_density_le_1"].to_numpy(),
            width=width,
            color=ACADEMIC_COLORS[idx % len(ACADEMIC_COLORS)],
            edgecolor="#2f2f2f",
            linewidth=0.6,
            hatch=ACADEMIC_HATCHES[idx % len(ACADEMIC_HATCHES)],
            label=model,
        )
    ax3.set_title("(c) Fraction of low-tail groups across sampled transformer blocks")
    ax3.set_xlabel("Transformer block index")
    ax3.set_ylabel("Fraction (density multiplier <= 1)")
    ax3.set_xticks(x, [str(v) for v in block_ids])
    ax3.grid(axis="y", alpha=0.35)
    ax3.legend(fontsize=9)

    for idx, model in enumerate(models):
        case_layer = choose_case_layer(summary_df, model, args.slot, args.case_layer_mode)
        cur = raw_df[
            (raw_df["model"] == model)
            & (raw_df["slot"] == args.slot)
            & (raw_df["layer_idx"] == case_layer)
        ].sort_values("outlier_density_multiplier", ascending=False)
        ax4.plot(
            np.arange(len(cur)),
            cur["outlier_density_multiplier"].to_numpy(),
            linewidth=1.8,
            color=ACADEMIC_COLORS[idx % len(ACADEMIC_COLORS)],
            label=f"{model} (B{case_layer})",
        )
    ax4.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax4.axhline(args.strong_outlier_multiplier, color="gray", linestyle=":", linewidth=1.0)
    ax4.set_title(f"(d) Sorted group density multipliers of a representative {args.slot} slot")
    ax4.set_xlabel("Group rank")
    ax4.set_ylabel("Outlier-density multiplier")
    ax4.grid(alpha=0.35)
    ax4.legend(fontsize=8)

    fig.suptitle(
        f"Sampled-layer group-wise activation distribution statistics (group size = {args.group_size})",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"fig3_3_activation_g{args.group_size}_distribution_summary.png"
    pdf_path = output_dir / f"fig3_3_activation_g{args.group_size}_distribution_summary.pdf"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def render_slot_bar_chart(summary_df: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    summary_df = summary_df[summary_df["group_size"] == args.group_size].copy()
    bar_df = (
        summary_df.groupby(["model", "slot"], as_index=False)[
            ["fraction_groups_density_gt_strong", "fraction_groups_density_le_1"]
        ]
        .mean()
        .sort_values(["model", "slot"])
    )
    models = sorted(bar_df["model"].unique().tolist())
    slots = ["qkv", "o", "gate_up", "down"]
    x = np.arange(len(slots))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for ax, metric, title in [
        (
            axes[0],
            "fraction_groups_density_gt_strong",
            f"Mean fraction of strong outlier-concentrated groups (> {args.strong_outlier_multiplier:g})",
        ),
        (
            axes[1],
            "fraction_groups_density_le_1",
            "Mean fraction of low-tail groups (<= baseline density)",
        ),
    ]:
        for idx, model in enumerate(models):
            cur = bar_df[bar_df["model"] == model].set_index("slot").reindex(slots)
            ax.bar(
                x + idx * width - width,
                cur[metric].to_numpy(),
                width=width,
                color=ACADEMIC_COLORS[idx % len(ACADEMIC_COLORS)],
                edgecolor="#2f2f2f",
                linewidth=0.6,
                hatch=ACADEMIC_HATCHES[idx % len(ACADEMIC_HATCHES)],
                label=model,
            )
        ax.set_xticks(x, slots)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.35)
    axes[0].set_ylabel("Fraction")
    axes[1].legend(fontsize=9)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / f"fig3_3_activation_g{args.group_size}_slot_bar.png"
    pdf_path = output_dir / f"fig3_3_activation_g{args.group_size}_slot_bar.pdf"
    fig.savefig(fig_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_academic_style()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    raw_df = pd.read_csv(input_dir / "group_activation_distribution_raw.csv")
    summary_df = pd.read_csv(input_dir / "group_activation_distribution_layer_summary.csv")
    render_summary_figure(raw_df, summary_df, args, output_dir)
    render_slot_bar_chart(summary_df, args, output_dir)
    print(f"[DONE] Figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
