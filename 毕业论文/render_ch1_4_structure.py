import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm, patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def load_font(weight="normal"):
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            return fm.FontProperties(fname=path, weight=weight)
    raise FileNotFoundError("No usable Chinese font found in C:\\Windows\\Fonts")


FP = load_font()
FP_B = load_font("bold")


def add_text(ax, x, y, text, size, color, bold=False, ha="center", va="center", **kwargs):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        color=color,
        fontproperties=FP_B if bold else FP,
        **kwargs,
    )


def main():
    out_dir = os.path.join(os.getcwd(), "generated_figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "fig1_3_thesis_structure_academic.png")

    fig = plt.figure(figsize=(14, 8), dpi=180)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("#f7f9fc")

    ax.add_patch(
        FancyBboxPatch(
            (2, 3),
            96,
            94,
            boxstyle="round,pad=0.8,rounding_size=3",
            facecolor="#f7f9fc",
            edgecolor="#d7dee8",
            linewidth=1.2,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (4, 88),
            92,
            7.5,
            boxstyle="round,pad=0.6,rounding_size=2.4",
            facecolor="#1f3a5f",
            edgecolor="#1f3a5f",
            linewidth=0,
        )
    )
    add_text(ax, 50, 91.6, "本文整体组织结构与研究主线", 20, "white", bold=True)
    add_text(
        ax,
        50,
        86.8,
        "围绕“误差机理分析—group-wise旋转算法设计—系统实现与实验验证”逐步展开",
        10.8,
        "#4a5b73",
    )

    stage_specs = [
        dict(
            x=8,
            w=25,
            color="#2f5d8c",
            light="#eaf1f8",
            title="阶段一  问题提出与理论基础",
            subtitle="研究背景、相关工作与技术基础",
        ),
        dict(
            x=37.5,
            w=25,
            color="#2d6f6d",
            light="#eaf6f5",
            title="阶段二  误差分析与方法实现",
            subtitle="误差模式、分布统计与算法系统",
        ),
        dict(
            x=67,
            w=25,
            color="#8a5a1e",
            light="#fbf4e8",
            title="阶段三  实验验证与总结展望",
            subtitle="性能分析、消融研究与结论归纳",
        ),
    ]

    for spec in stage_specs:
        ax.add_patch(
            FancyBboxPatch(
                (spec["x"], 18),
                spec["w"],
                62,
                boxstyle="round,pad=0.8,rounding_size=2.5",
                facecolor=spec["light"],
                edgecolor=spec["color"],
                linewidth=1.4,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (spec["x"] + 0.7, 74.5),
                spec["w"] - 1.4,
                5.5,
                boxstyle="round,pad=0.4,rounding_size=1.8",
                facecolor=spec["color"],
                edgecolor=spec["color"],
                linewidth=0,
            )
        )
        add_text(ax, spec["x"] + spec["w"] / 2, 77.2, spec["title"], 12.2, "white", bold=True)
        add_text(ax, spec["x"] + spec["w"] / 2, 71.6, spec["subtitle"], 9.2, "#506070")

    chapters = [
        dict(
            x=10.3,
            y=55,
            w=20.4,
            h=12,
            edge="#2f5d8c",
            no="第1章",
            title="前言",
            body="课题背景、研究现状\n研究内容与创新点",
        ),
        dict(
            x=10.3,
            y=33,
            w=20.4,
            h=12,
            edge="#2f5d8c",
            no="第2章",
            title="相关理论与技术基础",
            body="Transformer结构、PTQ基础\nNVFP4/MXFP4与旋转量化原理",
        ),
        dict(
            x=39.8,
            y=55,
            w=20.4,
            h=12,
            edge="#2d6f6d",
            no="第3章",
            title="误差来源与分布统计分析",
            body="group-wise误差模式\n权重/激活统计与设计启示",
        ),
        dict(
            x=39.8,
            y=33,
            w=20.4,
            h=12,
            edge="#2d6f6d",
            no="第4章",
            title="group-wise旋转算法与系统实现",
            body="搜索空间、目标函数、导出\n部署接入与自动化评测",
        ),
        dict(
            x=69.3,
            y=55,
            w=20.4,
            h=12,
            edge="#8a5a1e",
            no="第5章",
            title="实验结果与分析",
            body="整体效果、消融实验\n格式差异与部署开销分析",
        ),
        dict(
            x=69.3,
            y=33,
            w=20.4,
            h=12,
            edge="#8a5a1e",
            no="第6章",
            title="总结与展望",
            body="工作总结、主要结论\n后续研究方向",
        ),
    ]

    for ch in chapters:
        patch = FancyBboxPatch(
            (ch["x"], ch["y"]),
            ch["w"],
            ch["h"],
            boxstyle="round,pad=0.5,rounding_size=2.1",
            facecolor="#ffffff",
            edgecolor=ch["edge"],
            linewidth=1.6,
        )
        patch.set_path_effects(
            [patheffects.withSimplePatchShadow(offset=(1.1, -1.1), alpha=0.12)]
        )
        ax.add_patch(patch)
        add_text(ax, ch["x"] + 1.2, ch["y"] + 9.2, ch["no"], 10.4, ch["edge"], bold=True, ha="left")
        add_text(ax, ch["x"] + ch["w"] / 2, ch["y"] + 6.7, ch["title"], 11.2, "#1f2d3d", bold=True)
        add_text(
            ax,
            ch["x"] + ch["w"] / 2,
            ch["y"] + 3.0,
            ch["body"],
            9.0,
            "#475569",
            linespacing=1.35,
        )

    for x1, x2 in [(33.3, 37.0), (62.8, 66.5)]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, 49),
                (x2, 49),
                arrowstyle="simple",
                mutation_scale=16,
                linewidth=0,
                color="#90a4b8",
                alpha=0.95,
            )
        )
    for cx in [20.5, 50.0, 79.5]:
        ax.add_patch(
            FancyArrowPatch(
                (cx, 53.5),
                (cx, 45.5),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.0,
                color="#a8b3c2",
            )
        )

    ax.add_patch(
        FancyBboxPatch(
            (8, 8),
            84,
            6.2,
            boxstyle="round,pad=0.5,rounding_size=1.8",
            facecolor="#ffffff",
            edgecolor="#d3dbe6",
            linewidth=1.0,
        )
    )
    add_text(
        ax,
        50,
        11.1,
        "研究逻辑：问题提出与理论奠基  →  误差机理分析  →  group-wise旋转算法设计与系统实现  →  实验验证与总结展望",
        10.5,
        "#334155",
        bold=True,
    )

    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(out_path)


if __name__ == "__main__":
    main()
