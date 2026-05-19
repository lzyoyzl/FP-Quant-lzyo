from pathlib import Path

from docx import Document


DOC_NAME = "欧阳照林-毕业论文.docx"
SECTION_TITLE = "3.4 本章小结"
NEW_SUMMARY = (
    "本章围绕 group 粒度下的统计特征与量化误差来源展开分析，逐步明确了后续旋转算法设计所需要回答的核心问题。首先，通过比较全层旋转与局部旋转对局部统计结构的影响，说明在 NVFP4/MXFP4 这类依赖共享尺度的量化格式下，旋转策略必须与 group 粒度相匹配。其次，结合多模型、多层和不同槽位的局部列块分布统计，指出 outlier 虽然普遍存在，但大多数 group 仍保持相对平稳的分布形态，因此旋转不应统一施加，而应建立在选择性应用的基础上。进一步地，通过对 group-wise 量化误差来源的分析，本文将典型误差概括为离群值挤压型、大值主导型和混合型三类，并指出不同误差模式对应的主导因素与潜在变换目标并不一致。上述分析表明，面向低比特量化的旋转设计不能停留在固定矩阵或统一策略层面，而应以 group 列块为基本决策单元，结合局部分布与误差来源进行定制化选择。基于这一结论，下一章将进一步介绍本文所提出的 group-wise 旋转算法与系统实现细节，包括搜索空间设计、目标函数构造、量化导出以及推理接入等关键步骤。"
)


def main() -> None:
    path = Path(DOC_NAME)
    doc = Document(str(path))
    paragraphs = list(doc.paragraphs)

    target_idx = None
    for i, para in enumerate(paragraphs):
        if para.text.strip() == SECTION_TITLE:
            target_idx = i + 1
            break

    if target_idx is None or target_idx >= len(paragraphs):
        raise RuntimeError("Could not locate section 3.4 summary paragraph.")

    paragraphs[target_idx].text = NEW_SUMMARY
    doc.save(str(path))
    print(f"SAVED:{path.resolve()}")


if __name__ == "__main__":
    main()
