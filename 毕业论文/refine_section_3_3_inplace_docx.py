from pathlib import Path

from docx import Document


DOC_NAME = "欧阳照林-毕业论文.docx"
START_TITLE = "3.3 group 粒度下的量化误差来源分析"
END_TITLE = "3.4 本章小结"


REPLACEMENTS = [
    ("Heading 2", "3.3 group 粒度下的量化误差来源分析"),
    (
        "Normal",
        "3.2 节已经表明，不同 group 列块在局部分布上存在明显差异，因此旋转不宜再以整层统一处理为默认前提。进一步的问题在于，这种分布差异是否会真实反映到量化误差上，以及不同 group 的误差究竟由什么局部结构主导。只有回答了这一问题，才能进一步说明为什么后续方法需要在 group 粒度上做选择性的旋转，并针对不同列块采用不同的变换。",
    ),
    (
        "Normal",
        "经典量化理论通常从更粗粒度的场景出发进行解释。以 channel 粒度为例，当一个极少数离群通道与大量正常通道共同决定共享尺度时，真正贡献主要量化误差的往往并不是离群通道本身，而是数量占绝对多数的 normal value channel。也就是说，离群值的作用在于抬升量化步长，而误差则主要累积在大量普通数值上。这一解释能够较好说明传统场景中的 outlier 问题，但当共享尺度缩小到 group 粒度之后，组内数值结构更加局部化，误差来源也不再必然服从同一规律。",
    ),
    ("Heading 3", "3.3.1 group 粒度下的量化误差来源归纳"),
    (
        "Normal",
        "为此，本文首先对若干抽样 Transformer block 的典型输入进行统计，分别记录每个 group 的 quantization error 与组内最大值，并比较二者之间的对应关系。统计结果显示，确有一部分 group 呈现出“最大值越突出、量化误差越高”的现象，这与经典结论一致；但与此同时，也存在一些 group 虽然最大值并不极端，却仍然具有较高量化误差，或者在最大值相近的情况下误差水平差异显著。这说明在 group 粒度下，仅凭最大值统计已不足以完整解释误差来源。",
    ),
    (
        "Normal",
        "【插图占位：此处插入三张“每个 group 的 quant error（红色）与最大值（蓝色）统计图”，分别对应三类典型情况】",
    ),
    ("Caption", "图3-4 典型 group 的量化误差与最大值统计关系（占位）"),
    (
        "Normal",
        "在上述观察基础上，本文进一步选取每个 case 中量化误差最大的 5 个 group，对组内数据进行更细粒度分析。具体做法是将组内数值按相对最大值归一化到 [0,1] 区间，再考察不同分位点区间对总量化误差的贡献。通过这种方式可以更直接地判断：误差究竟主要来自靠近零附近的大量普通值，还是来自高分位区域中的若干大值，抑或由二者共同作用形成。",
    ),
    (
        "Normal",
        "综合这些高误差 group 的分位点误差贡献，可以将 group 粒度下的量化误差概括为三类。第一类是离群值挤压型误差，此时主要误差集中在低分位区域，说明少数峰值抬高了组内步长，而大量普通值承担了主要误差；第二类是大值主导型误差，此时误差主要集中在高分位区域，表明问题不只在于单一 outlier，而在于多个较大值共同决定了表示稀疏性；第三类是混合型误差，即低分位与高分位区域同时贡献显著误差，反映出组内既存在普通值被挤压的问题，也存在大值簇自身表示不足的问题。这三类情况共同说明，group 粒度下的误差机制已经明显比经典 channel 粒度分析更为复杂。",
    ),
    (
        "Normal",
        "【插图占位：此处插入三张 top-5 高误差 group 的 zoom-in 分析图，分别对应离群值挤压型、大值主导型和混合型误差】",
    ),
    ("Caption", "图3-5 高误差 group 的分位点误差贡献分析（占位）"),
    (
        "Normal",
        "因此，从误差来源角度看，3.2 节所观察到的“不同 group 列块分布差异”并不仅仅是统计现象，而会进一步转化为真实的量化误差差异。更重要的是，这种差异并不总能由最大值大小单独解释，而需要结合组内数值在不同分位区间上的误差贡献来加以区分。由此可以得到一个更明确的结论：在 group 粒度下，后续变换设计不能假设所有列块面对的是同一种误差机制。",
    ),
    ("Heading 3", "3.3.2 不同误差来源下的变换启示"),
    (
        "Normal",
        "局部变换之所以可能改善量化误差，本质上在于它能够重新组织组内数值在不同方向上的投影关系。当一个 group 中存在明显主导方向时，适当的旋转可以削弱峰值对共享尺度的单边支配，从而缓解离群值挤压现象；而当误差来自多个较大值之间的相对关系时，变换的作用则更多体现在重新分配高分位数值的局部结构，使其在共享尺度下获得更协调的表示。",
    ),
    (
        "Normal",
        "但这并不意味着旋转在任何 group 上都会自然生效。如果一个 group 的分布本身已经较为均匀，各个方向上的投影差异较小，那么旋转并不一定能够找到更优方向，甚至可能破坏原有的平衡结构，使原本较稳定的列块反而引入额外误差。换言之，变换是否有效，取决于它是否真正对应了该 group 的主导误差来源；对于本就均匀的局部列块，保持 identity 反而可能是更稳妥的选择。",
    ),
    (
        "Normal",
        "【插图占位：此处插入 Transformation 示意图，用于说明不同误差来源下局部变换可能带来的不同效果】",
    ),
    ("Caption", "图3-6 不同误差来源下局部变换作用差异示意图（占位）"),
    (
        "Normal",
        "由此可以得到本节的最终结论：group 粒度下的旋转不应被理解为“统一矩阵对所有列块做同样处理”，而应建立在误差来源分析基础上进行定制化选择。对于离群值挤压型 group，更适合采用能够削弱单一峰值主导的变换；对于大值主导型 group，更应关注多个高分位数值之间的局部重组；而对于混合型或本身较为均匀的 group，则需要在变换收益与结构扰动之间进行权衡，必要时保留 identity 选项。",
    ),
    ("Caption", "表3-2 三类 group 粒度量化误差来源及其变换启示（占位）"),
    (
        "Normal",
        "【表格占位：此处插入离群值挤压型、大值主导型、混合型误差的局部特征、主要误差来源及适宜变换方向归纳表】",
    ),
]


def main() -> None:
    doc_path = Path(DOC_NAME)
    doc = Document(str(doc_path))
    paragraphs = list(doc.paragraphs)

    start_idx = None
    end_idx = None
    for i, para in enumerate(paragraphs):
        text = para.text.strip()
        if text == START_TITLE:
            start_idx = i
        elif text == END_TITLE and start_idx is not None:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        raise RuntimeError("Could not locate section 3.3 boundaries.")

    target = paragraphs[start_idx:end_idx]
    if len(target) != len(REPLACEMENTS):
        raise RuntimeError(
            f"Paragraph count mismatch: section has {len(target)} paragraphs, replacement has {len(REPLACEMENTS)}."
        )

    for para, (style_name, text) in zip(target, REPLACEMENTS):
        para.style = doc.styles[style_name]
        para.text = text

    doc.save(str(doc_path))
    print(f"SAVED:{doc_path.resolve()}")


if __name__ == "__main__":
    main()
