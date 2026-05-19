from pathlib import Path

from docx import Document


DOC_PATH = Path(r"\\wsl.localhost\Ubuntu-22.04\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx")
SECTION_TITLE = "2.4　本章小结"
NEXT_TITLE = "第3章 量化误差来源与权重/激活分布统计分析"

NEW_TEXT = (
    "本章围绕后续方法构建所需的理论基础，系统梳理了后训练低比特量化、微缩放量化格式与旋转辅助量化三方面内容。"
    "首先，在后训练低比特量化基础部分，本文从量化映射、反量化过程及典型误差来源出发，说明了共享尺度、舍入与截断等因素对量化精度的影响，"
    "并结合典型 PTQ 方法分析了低比特条件下精度保持的基本思路。其次，在 NVFP4 与 MXFP4 微缩放量化格式部分，本文总结了两类格式在 group 粒度、"
    "尺度表示方式与开销特征上的差异，指出 microscaling 机制虽然提升了局部尺度适配能力，但也使量化结果更加依赖 group 内部的数据分布。最后，"
    "在旋转辅助量化原理部分，本文从线性映射等价性出发，阐明了旋转通过重塑局部能量分布、缓解 outlier 主导效应以改善共享尺度匹配质量的基本机理，"
    "并对几类常见旋转方法的性质与适用场景进行了归纳。上述分析表明，低比特浮点量化的实际误差不仅受量化格式本身约束，还与权重和激活在局部 group 中的统计特征密切相关。"
    "因此，下一章将在此基础上进一步围绕量化误差来源以及权重/激活分布统计展开分析，为后续 group-wise 旋转算法的设计提供更直接的依据。"
)


def main() -> None:
    doc = Document(str(DOC_PATH))
    paragraphs = doc.paragraphs

    start_idx = None
    next_idx = None
    for idx, para in enumerate(paragraphs):
        text = para.text.strip()
        if text == SECTION_TITLE:
            start_idx = idx
        elif start_idx is not None and text == NEXT_TITLE:
            next_idx = idx
            break

    if start_idx is None or next_idx is None or next_idx <= start_idx:
        raise RuntimeError("Failed to locate section 2.4 boundaries.")

    if next_idx - start_idx == 1:
        body_para = paragraphs[start_idx].insert_paragraph_before("")
        # move created paragraph after section title by swapping text later is cumbersome,
        # so use existing next heading's preceding paragraph slot instead when available
        raise RuntimeError("Unexpected empty section 2.4; please create one body paragraph first.")

    body_para = paragraphs[start_idx + 1]
    body_para.text = NEW_TEXT

    for idx in range(next_idx - 1, start_idx + 1, -1):
        p = paragraphs[idx]._element
        p.getparent().remove(p)

    doc.save(str(DOC_PATH))
    print(f"SAVED:{DOC_PATH}")


if __name__ == "__main__":
    main()
