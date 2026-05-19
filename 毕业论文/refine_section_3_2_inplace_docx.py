from pathlib import Path

from docx import Document


DOC_PATH = Path(r"\\wsl.localhost\Ubuntu-22.04\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx")
START_TITLE = "3.2 局部列块分布统计与选择性旋转"
END_TITLE = "3.3 group 粒度下的量化误差来源分析"


def has_drawing(paragraph) -> bool:
    return any(el.tag.endswith("}drawing") for el in paragraph._element.iter())


def main():
    doc = Document(str(DOC_PATH))
    paragraphs = doc.paragraphs

    start_idx = next(i for i, p in enumerate(paragraphs) if p.text.strip() == START_TITLE)
    end_idx = next(i for i, p in enumerate(paragraphs) if p.text.strip() == END_TITLE)

    section_paras = paragraphs[start_idx + 1:end_idx]
    text_paras = [p for p in section_paras if not has_drawing(p)]

    replacement_texts = [
        "为进一步判断哪些局部列块值得旋转，本文选取 llama-2-7b-hf、llama-3-8b-hf 和 Qwen3-8B 作为统计对象，在多个抽样 Transformer block 上对 qkv、o、gate_up 和 down 槽位输入的 group-wise 激活分布进行统计。统计分别采用 group size 为 16 和 32 的两种粒度，对应更细和更粗的局部列块划分方式，以考察不同粒度下局部异常与常规列块的分布特征。",
        "图3-3(a)、(b)给出了 group size 为 16 和 32 时抽样 Transformer block 的综合统计结果，重点展示强 outlier 富集 group 比例、group outlier-density multiplier 的累积分布，以及代表性 qkv 槽位中各 group 的排序情况；图3-3(c)、(d)则进一步比较了不同槽位上强 outlier 富集 group 与低尾 group 的平均比例，用于考察槽位差异及统计结论在不同 group 粒度下的一致性。",
        "（a）group size=16 条件下抽样 Transformer block 的 group-wise 激活分布综合统计图",
        "（b）group size=32 条件下抽样 Transformer block 的 group-wise 激活分布综合统计图",
        "（c）group size=16 条件下不同槽位的平均 group 比例统计图",
        "（d）group size=32 条件下不同槽位的平均 group 比例统计图",
        "图3-3 不同 group 粒度下局部列块激活分布统计结果",
        "从图3-3(a)、(b)可以看到，无论在 group size=16 还是 32 的设置下，三类模型在多个抽样 Transformer block 上都存在非零比例的强 outlier 富集 group，说明局部异常并非只出现在个别模型或少数特殊层中，而是在大语言模型内部广泛存在。与此同时，不同模型在不同 block 上的比例虽然存在一定波动，但整体量级相近，表明这种局部异常具有较强的普遍性。",
        "但更值得注意的是，图3-3(a)、(b)中的 CDF 曲线都在靠近 1 的区域快速上升，并在较小 multiplier 范围内迅速逼近 1；代表性 qkv 槽位的排序曲线也显示，只有排名靠前的少数 group 具有明显偏高的 outlier 密度，而绝大多数 group 很快回落到接近基线甚至低于基线的区域。这说明局部异常主要集中于少数列块，大多数 group 的局部分布实际上仍然相对均匀。",
        "图3-3(c)、(d)进一步从槽位角度印证了这一判断。不同槽位虽然存在差异，例如 qkv 和 gate_up 槽位中强 outlier 富集 group 的平均比例相对更高，而部分槽位的低尾 group 比例略低，但在两种 group 粒度下，各槽位的低尾 group 仍占据主体，且这种趋势在三类模型中保持一致。这表明 group 粒度的统计规律并非偶然结果，也说明不同结构功能的线性层在局部异常程度上并不完全相同。",
        "综合上述结果可以得到更有针对性的结论：旋转应当建立在局部分布统计基础上，有选择地施加于确有必要的 group。强 outlier 富集 group 的广泛存在说明旋转具有现实必要性，而大多数列块的相对均匀性则进一步说明旋转不能采取“一刀切”的方式。更合理的策略应当是在 group 粒度上识别局部异常程度，并仅对少数确有必要的列块施加变换，以尽量避免对原本平滑的常规列块引入额外扰动。",
    ]

    if len(text_paras) < len(replacement_texts):
        raise RuntimeError(f"Section 3.2 text paragraph count mismatch: have {len(text_paras)}, need {len(replacement_texts)}")

    for para, text in zip(text_paras, replacement_texts):
        para.text = text

    doc.save(str(DOC_PATH))
    print(f"SAVED:{DOC_PATH}")


if __name__ == "__main__":
    main()
