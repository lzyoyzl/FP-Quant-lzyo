from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.font_manager import FontProperties


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "generated_figures"
OUT_DIR.mkdir(exist_ok=True)


def get_font():
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return FontProperties(fname=str(path))
    return None


FONT = get_font()


def text(ax, x, y, s, size=12, weight="normal", color="#17324d", ha="center", va="center", **kwargs):
    ax.text(
        x,
        y,
        s,
        fontproperties=FONT,
        fontsize=size,
        fontweight=weight,
        color=color,
        ha=ha,
        va=va,
        **kwargs,
    )


def rounded(ax, xy, w, h, fc, ec="#17324d", lw=1.4, r=0.025):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.01,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, color="#28567c", width=1.8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=width,
            color=color,
            connectionstyle="arc3,rad=0.0",
        )
    )


def canvas(w=14, h=8):
    fig, ax = plt.subplots(figsize=(w, h), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def fig_2_1():
    fig, ax = canvas(14.6, 8.3)
    ax.add_patch(Rectangle((0.03, 0.9), 0.94, 0.07, facecolor="#f3f7fb", edgecolor="#d2dee9", linewidth=1.2))
    text(ax, 0.05, 0.935, "Transformer Block 结构与本文关注的线性层位置", size=17.5, weight="bold", ha="left")
    text(ax, 0.95, 0.935, "相关理论与技术基础", size=12.8, ha="right", color="#28567c")

    rounded(ax, (0.06, 0.77), 0.14, 0.08, "#edf4fb")
    text(ax, 0.13, 0.81, "输入隐藏状态 X", size=12.5, weight="bold")
    arrow(ax, (0.20, 0.81), (0.27, 0.81))

    rounded(ax, (0.27, 0.75), 0.13, 0.11, "#f4f8ef")
    text(ax, 0.335, 0.81, "RMSNorm / LN", size=12.2, weight="bold")
    text(ax, 0.335, 0.775, "归一化稳定数值范围", size=9.8, color="#44617c")

    arrow(ax, (0.40, 0.81), (0.47, 0.81))
    rounded(ax, (0.47, 0.67), 0.18, 0.22, "#eef5fb")
    ax.add_patch(Rectangle((0.47, 0.84), 0.18, 0.05, facecolor="#17324d", edgecolor="none"))
    text(ax, 0.56, 0.865, "自注意力分支", size=12.3, weight="bold", color="white")

    sub_w = 0.045
    sub_h = 0.065
    xs = [0.495, 0.5475, 0.60]
    labels = ["q_proj", "k_proj", "v_proj"]
    for x, lbl in zip(xs, labels):
        rounded(ax, (x, 0.74), sub_w, sub_h, "#ffffff", ec="#cfd8e2", lw=1.0, r=0.015)
        text(ax, x + sub_w / 2, 0.772, lbl, size=9.8, weight="bold")
        arrow(ax, (x + sub_w / 2, 0.74), (0.56, 0.705), width=1.2)
    rounded(ax, (0.505, 0.675), 0.11, 0.045, "#ffffff", ec="#cfd8e2", lw=1.0, r=0.015)
    text(ax, 0.56, 0.697, "Attention(Q,K,V)", size=10.0, weight="bold")
    rounded(ax, (0.512, 0.605), 0.096, 0.045, "#ffffff", ec="#cfd8e2", lw=1.0, r=0.015)
    text(ax, 0.56, 0.627, "o_proj", size=10.0, weight="bold")

    arrow(ax, (0.56, 0.605), (0.56, 0.55))
    rounded(ax, (0.49, 0.50), 0.14, 0.05, "#f7f1fb")
    text(ax, 0.56, 0.525, "残差相加", size=10.5, weight="bold")

    arrow(ax, (0.56, 0.50), (0.56, 0.45))
    rounded(ax, (0.42, 0.25), 0.28, 0.17, "#fbf5ee")
    ax.add_patch(Rectangle((0.42, 0.37), 0.28, 0.05, facecolor="#7a4e1d", edgecolor="none"))
    text(ax, 0.56, 0.395, "前馈网络分支", size=12.3, weight="bold", color="white")
    rounded(ax, (0.445, 0.30), 0.07, 0.045, "#ffffff", ec="#d7dfe7", lw=1.0, r=0.015)
    rounded(ax, (0.525, 0.30), 0.07, 0.045, "#ffffff", ec="#d7dfe7", lw=1.0, r=0.015)
    rounded(ax, (0.605, 0.30), 0.07, 0.045, "#ffffff", ec="#d7dfe7", lw=1.0, r=0.015)
    text(ax, 0.48, 0.323, "gate_proj", size=9.3, weight="bold")
    text(ax, 0.56, 0.323, "up_proj", size=9.3, weight="bold")
    text(ax, 0.64, 0.323, "down_proj", size=9.3, weight="bold")
    text(ax, 0.56, 0.268, "本文后续量化与旋转优化的核心对象", size=10.0, color="#6d4a24")

    arrow(ax, (0.20, 0.81), (0.49, 0.525), color="#90a4b8", width=1.0)
    arrow(ax, (0.56, 0.45), (0.56, 0.42), width=1.5)
    arrow(ax, (0.56, 0.25), (0.56, 0.18), width=1.5)

    rounded(ax, (0.45, 0.12), 0.22, 0.05, "#17324d", ec="#17324d", lw=1.0, r=0.018)
    text(ax, 0.56, 0.145, "输出隐藏状态", size=11.8, weight="bold", color="white")

    notes = [
        (0.77, 0.73, "q / k / v：\n决定注意力计算的查询、键和值表示"),
        (0.77, 0.54, "o_proj：\n完成多头信息融合并回到隐藏空间"),
        (0.77, 0.32, "gate / up / down：\n控制 MLP 扩展、非线性调制与回投影"),
    ]
    for x, y, s in notes:
        rounded(ax, (x, y - 0.055), 0.16, 0.10, "#ffffff", ec="#d5dee7", lw=1.0, r=0.015)
        text(ax, x + 0.08, y - 0.005, s, size=9.6, color="#2f4357", linespacing=1.35)

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT_DIR / "fig2_1_transformer_block.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_2_2():
    fig, ax = canvas(14.8, 7.8)
    ax.add_patch(Rectangle((0.03, 0.9), 0.94, 0.07, facecolor="#f3f7fb", edgecolor="#d2dee9", linewidth=1.2))
    text(ax, 0.05, 0.935, "低比特量化与旋转辅助的等价部署关系", size=17.5, weight="bold", ha="left")
    text(ax, 0.95, 0.935, "PTQ / Microscaling / Rotation", size=12.8, ha="right", color="#28567c")

    xs = [0.06, 0.28, 0.50, 0.72]
    titles = ["高精度张量", "正交旋转 / 重参数化", "分组量化与共享 scale", "推理等价执行"]
    subtitles = [
        "权重 / 激活存在重尾分布\n少量通道可能主导尺度",
        "改变坐标基，削弱峰值主导\n保留线性映射等价性",
        "NVFP4 / MXFP4 在 group 内共享 scale\n块内统计稳定性决定精度上限",
        "激活侧前向旋转\n权重侧逆变换折叠",
    ]
    colors = ["#eef5fb", "#f4f8ef", "#fbf5ee", "#f7f1fb"]
    accents = ["#17324d", "#2f5a32", "#7a4e1d", "#5c476f"]

    for i, x in enumerate(xs):
        rounded(ax, (x, 0.46), 0.18, 0.28, colors[i], ec=accents[i], lw=1.4)
        ax.add_patch(Rectangle((x, 0.69), 0.18, 0.05, facecolor=accents[i], edgecolor="none"))
        text(ax, x + 0.09, 0.715, titles[i], size=12.0, weight="bold", color="white")
        text(ax, x + 0.09, 0.57, subtitles[i], size=10.3, color="#2f4357", linespacing=1.4)
        if i < len(xs) - 1:
            arrow(ax, (x + 0.18, 0.60), (xs[i + 1] - 0.015, 0.60), color="#28567c", width=1.8)

    rounded(ax, (0.12, 0.18), 0.25, 0.12, "#ffffff", ec="#cfd8e2", lw=1.0, r=0.018)
    text(ax, 0.245, 0.245, "误差来源", size=12.0, weight="bold")
    text(ax, 0.245, 0.205, "舍入误差 / 截断误差 /\n组内尺度失配 / 层间传播", size=10.1, color="#2f4357", linespacing=1.35)

    rounded(ax, (0.40, 0.18), 0.22, 0.12, "#ffffff", ec="#cfd8e2", lw=1.0, r=0.018)
    text(ax, 0.51, 0.245, "旋转作用", size=12.0, weight="bold")
    text(ax, 0.51, 0.205, "扩散局部峰值\n改善 group 内统计均衡性", size=10.1, color="#2f4357", linespacing=1.35)

    rounded(ax, (0.65, 0.18), 0.23, 0.12, "#ffffff", ec="#cfd8e2", lw=1.0, r=0.018)
    text(ax, 0.765, 0.245, "本文关注点", size=12.0, weight="bold")
    text(ax, 0.765, 0.205, "不同线性层与不同格式下\n何种旋转更适配", size=10.1, color="#2f4357", linespacing=1.35)

    arrow(ax, (0.37, 0.24), (0.40, 0.24), color="#7089a0", width=1.2)
    arrow(ax, (0.62, 0.24), (0.65, 0.24), color="#7089a0", width=1.2)

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT_DIR / "fig2_2_rotation_quant_pipeline.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_2_1()
    fig_2_2()
