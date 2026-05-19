import sys
from pathlib import Path

from docx import Document

from rewrite_chapter4_docx import (
    delete_paragraph,
    insert_caption_before,
    insert_equation_table_before,
    insert_heading_before,
    insert_table_before,
    insert_text_paragraph_before,
    m_frac,
    m_sub,
    m_subsup,
    m_sup,
    remove_tables_between,
)


DEFAULT_DOC_PATH = Path(r"Z:\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx")
START_TITLE = "4.2 group-wise 旋转搜索空间设计"
END_TITLE = "4.3 搜索目标函数与算法流程"


def rebuild_section_4_2(doc: Document, anchor):
    insert_heading_before(anchor, "4.2.1 变换粒度与作用范围", "Heading 3")
    insert_text_paragraph_before(
        anchor,
        "从代码实现看，旋转搜索空间首先受量化格式约束。主程序在解析参数后会对 NVFP4 与 MXFP4 的 group 大小进行统一规范："
        "NVFP4 强制使用 weight/activation group_size=16，MXFP4 强制使用 weight/activation group_size=32。"
        "这意味着后续所有候选变换都不是以整层矩阵为基本单元，而是以量化格式已经确定好的局部列块为作用对象。",
        "Normal",
    )
    insert_caption_before(anchor, "表4-1 不同量化格式的 group size 约束")
    insert_table_before(
        anchor,
        doc,
        [
            ["量化格式", "weight group_size", "activation group_size", "scale precision"],
            ["NVFP4", "16", "16", "E4M3"],
            ["MXFP4", "32", "32", "E8M0"],
        ],
    )
    insert_equation_table_before(
        anchor,
        [
            m_sub("T", "slot"), " = blkdiag(",
            m_sub("T", "1"), ", ", m_sub("T", "2"), ", … , ", m_sub("T", "G"),
            ")",
        ],
        "（4-3）",
        doc,
    )
    insert_text_paragraph_before(
        anchor,
        "式（4-3）刻画了当前实现中 group-wise 变换的总体组织方式。"
        "对某个输入槽位而言，整层输入方向上的旋转并不是由一个统一矩阵完成，而是由各个 group 列块对应的局部矩阵按块对角形式拼接而成。"
        "代码中的 MixedGroupTransform 正是围绕这一结构实现的：它将每个列块选中的前向矩阵与逆转置矩阵分别堆叠保存，"
        "并在初始化时显式检查线性等价关系，以保证激活侧前向旋转与权重侧逆变换折叠严格对应。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "在作用范围上，build_block_input_transforms 会围绕每个 Transformer block 构造四个搜索任务："
        "q_proj、k_proj 与 v_proj 共用 qkv 槽位；o_proj 单独对应 o 槽位；gate_proj 与 up_proj 共用 gate_up 槽位；"
        "down_proj 则对应 down 槽位。这样做的直接原因是这些线性层在输入统计和功能角色上具有不同的局部结构，"
        "同时又存在可共享搜索的自然分组。",
        "Normal",
    )
    insert_caption_before(anchor, "表4-2 四类输入槽位的搜索对象与作用范围")
    insert_table_before(
        anchor,
        doc,
        [
            ["槽位", "共享线性层", "输入特征维度来源", "搜索对象"],
            ["qkv", "q_proj / k_proj / v_proj", "hidden_size", "共享的 group-wise 输入变换"],
            ["o", "o_proj", "hidden_size", "独立的 group-wise 输入变换"],
            ["gate_up", "gate_proj / up_proj", "hidden_size", "共享的 group-wise 输入变换"],
            ["down", "down_proj", "intermediate_size", "独立的 group-wise 输入变换"],
        ],
    )
    insert_text_paragraph_before(
        anchor,
        "结合第三章的误差来源分析，这种设计的意义在于：算法不再默认所有局部列块都需要同一种旋转，而是允许“在同一槽位内，不同 group 面向不同误差模式选择不同候选矩阵”。"
        "因此，搜索空间的设计本身就已经将“选择性旋转”写入了实现结构，而不是在实验阶段再做额外修补。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "【伪代码占位：展示 build_block_input_transforms 如何围绕 qkv、o、gate_up、down 四个槽位构造 group-wise 搜索任务，并将搜索结果组装为 MixedGroupTransform。】",
        "Normal",
    )

    insert_heading_before(anchor, "4.2.2 Identity 变换", "Heading 3")
    insert_equation_table_before(
        anchor,
        [m_sub("T", "id"), " = ", m_sub("I", "g")],
        "（4-4）",
        doc,
    )
    insert_text_paragraph_before(
        anchor,
        "Identity 变换对应式（4-4），即保持当前 group 列块的原始坐标方向不变。"
        "虽然它在形式上是最简单的候选矩阵，但在当前实现中并不是可有可无的占位项，而是搜索空间中不可缺少的一类基线候选。"
        "第三章已经表明，许多局部列块本身分布较为平滑，量化误差并不一定由明显的方向性异常主导；"
        "对这类 group 而言，额外施加旋转反而可能破坏原有均衡结构，使本来较低的尾部误差被重新放大。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "从代码逻辑看，只要某个 group 在给定目标函数下由 Identity 取得最小评分，系统就会直接保留其原始方向。"
        "这意味着 Identity 不是“什么都不做”的消极选项，而是对“旋转未必有效”这一结论的实现化表达。"
        "它尤其适合用于误差来源较弱、尾部不突出的低风险列块，以避免对均匀分布做不必要的结构性扰动。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "【伪代码占位：展示 IdentityTransform 在搜索阶段与其他候选矩阵同等参与打分，并在最优时直接作为该 group 的保留变换。】",
        "Normal",
    )

    insert_heading_before(anchor, "4.2.3 Householder 变换", "Heading 3")
    insert_equation_table_before(
        anchor,
        [
            m_sub("H", "g"), "(", "v", ") = ",
            m_sub("I", "g"), " - ",
            m_frac(["2", " ", "v", " ", m_sup("v", "T")], [m_sup("v", "T"), " ", "v"]),
        ],
        "（4-5）",
        doc,
    )
    insert_equation_table_before(
        anchor,
        [
            m_sup(m_sub("H", "g"), "T"), " ", m_sub("H", "g"), " = ", m_sub("I", "g"),
            ",    ",
            m_sup(m_sub("H", "g"), "-1"), " = ", m_sub("H", "g"),
        ],
        "（4-6）",
        doc,
    )
    insert_text_paragraph_before(
        anchor,
        "Householder 变换是当前搜索空间中最具“方向自适应潜力”的候选之一。"
        "对任意非零向量 v，式（4-5）给出了对应的反射矩阵构造方式。"
        "该矩阵以向量 v 所张成的方向为中心，对整个局部子空间做一次关于超平面的镜像反射；式（4-6）则表明它既是正交矩阵，又满足自逆性质，"
        "因此能够直接用于当前“激活侧前向旋转、权重侧逆变换折叠”的部署框架。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "从具体实现看，HouseholderTransform 会先随机初始化一个长度为 group_size 的向量 v，"
        "然后构造单个 group 反射块，再将这一反射块按列块重复为 block-diagonal 形式。"
        "也就是说，代码并不是显式学习一个满秩矩阵，而是通过一个方向向量生成局部反射矩阵。"
        "这一构造方式的好处在于参数形式紧凑，同时仍能明显改变组内主导方向与坐标轴之间的夹角关系。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "若结合第三章的误差来源分析，Householder 变换尤其适合处理那些“存在明显主导方向，但又不适合简单均匀扩散”的列块。"
        "对离群值挤压型误差而言，反射可以改变峰值在原始坐标系上的投影位置，弱化单一方向对共享尺度的独占；"
        "对大值主导型误差而言，Householder 则更像是一种低自由度的方向重排，使多个较大分量在新的基下获得更协调的局部分布。"
        "因此，它可以看作是介于固定基变换与完全自由正交矩阵之间的一种折中选择。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "在矩阵形式上，如果取二维示例向量 v=[1,1]^T，则可得到一个关于对角方向的反射矩阵。"
        "虽然实际实现中的 group_size 远大于 2，但这一例子有助于理解其本质：Householder 并不是对能量做简单平均扩散，"
        "而是通过关于特定方向的反射重新组织局部投影结构。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "【伪代码占位：展示 HouseholderTransform 中随机生成向量 v、构造单 group 反射块、按 group 重复形成 block-diagonal 矩阵并用于前向/逆向计算的实现流程。】",
        "Normal",
    )

    insert_heading_before(anchor, "4.2.4 Hadamard 类变换", "Heading 3")
    insert_equation_table_before(
        anchor,
        [
            m_sub("T", "had"), " = ", m_frac("1", "√g"), " ", m_sub("H", "g"),
            ",    ",
            m_sub("H", "2g"), " = ",
            "[[", m_sub("H", "g"), ", ", m_sub("H", "g"), "], [", m_sub("H", "g"), ", -", m_sub("H", "g"), "]]",
        ],
        "（4-7）",
        doc,
    )
    insert_text_paragraph_before(
        anchor,
        "Hadamard 变换是当前实现中最典型的结构化正交候选。"
        "式（4-7）给出了归一化 Hadamard 矩阵的定义及其递归构造方式。"
        "在代码中，HadamardTransform 并不显式构造整层大矩阵，而是按 group_size 将输入重排后调用快速 Hadamard 变换，"
        "再乘以归一化系数 1/√g。这样既保证了数值形式与理论一致，也减少了在搜索阶段反复构造大规模稠密矩阵的开销。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "从误差缓解角度看，Hadamard 类变换最直接的作用是“扩散”。"
        "对于离群值挤压型 group，单一大值往往会抬升整个列块的量化步长，而 Hadamard 变换会将该方向上的集中能量分散到多个基向量上，"
        "从而降低共享尺度被单一峰值支配的程度。"
        "也正因为这种扩散能力，Hadamard 往往能有效缓解 A 类误差，但若直接作用于本就均匀的 group，"
        "则可能把原本平稳的局部结构重新打散，带来额外量化扰动，这也是第三章所强调“不能一刀切旋转”的原因之一。",
        "Normal",
    )
    insert_equation_table_before(
        anchor,
        [
            m_sub("T", "gsr"), " = ", m_frac("1", "√g"), " ", m_sub("H", "g"), " ", m_sub("P", "gsr"),
        ],
        "（4-8）",
        doc,
    )
    insert_text_paragraph_before(
        anchor,
        "在 Hadamard 的基础上，代码还实现了 GSR 变换。"
        "其核心做法是先生成 Hadamard 基，再按照列向量的符号变化次数对列进行重排，"
        "由此得到式（4-8）中的重排序矩阵 P_gsr。"
        "与固定顺序的 Hadamard 相比，GSR 仍保持结构化、正交和自逆等优点，但在组内方向顺序上进行了再组织，"
        "因此它更像是“以 Hadamard 为底座的局部方向重排”。在某些列块上，这种重排能够在维持扩散特性的同时，"
        "给出比标准 Hadamard 更贴近局部分布结构的旋转方式。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "【伪代码占位：展示 HadamardTransform 的分块快速变换实现，以及 GSRTransform 中如何统计列符号变化次数并构造重排序矩阵。】",
        "Normal",
    )

    insert_heading_before(anchor, "4.2.5 其他候选正交变换", "Heading 3")
    insert_text_paragraph_before(
        anchor,
        "除 Householder 和 Hadamard 类变换外，默认候选集合中还包括 DCT 与 DST；项目中同时保留了 FastFood 的实现，"
        "但默认搜索空间并未启用该分支。相比 Hadamard 这类以符号翻转和能量扩散为主的结构化变换，"
        "DCT 与 DST 更强调基向量的平滑频率结构，因此更适合处理那些“并非由单一极端峰值主导，而是存在稳定局部相关性”的列块。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "在实现上，DCTTransform 与 DSTransform 都是先构造 group_size 维的 type-II 正交基矩阵，"
        "再按 group 重复为 block-diagonal 形式。也就是说，它们的构造逻辑更接近“显式生成局部正交基”，"
        "而不是像 Hadamard 那样依赖快速递推变换。这样的差异使得 DCT/DST 在数值结构上更平滑，也更容易对局部相关分量做重新组织。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "对第三章中提到的大值主导型或混合型误差而言，DCT 与 DST 的意义在于："
        "它们不一定像 Hadamard 那样强烈地打散峰值，而是更倾向于改变局部相关方向与坐标轴之间的匹配关系。"
        "如果某个 group 的误差并不是来自单点离群值，而是来自多个较大分量在原始基下共同形成的不利结构，"
        "那么这种平滑基变换往往更有可能在不造成过度扩散的前提下改善量化分布。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "FastFood 变换在代码中也给出了结构化随机矩阵实现，但由于其依赖更复杂的随机因子与多重矩阵乘法，"
        "当前默认搜索空间仍然聚焦于 Identity、Hadamard、DCT、DST、GSR 与 Householder 这六类候选。"
        "从工程角度看，这样的选择兼顾了候选多样性、数值稳定性以及实现复杂度，也更便于后续导出与推理接入。",
        "Normal",
    )
    insert_text_paragraph_before(
        anchor,
        "【伪代码占位：展示 DCT/DST 的局部正交基构造流程，以及 FastFood 作为可选扩展候选时的结构化随机矩阵生成步骤。】",
        "Normal",
    )


def main():
    doc_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOC_PATH
    if not doc_path.is_absolute():
        doc_path = (Path.cwd() / doc_path).resolve()

    doc = Document(str(doc_path))
    paragraphs = list(doc.paragraphs)

    start_idx = None
    end_idx = None
    for i, p in enumerate(paragraphs):
        text = p.text.strip()
        if text == START_TITLE:
            start_idx = i
        elif start_idx is not None and text == END_TITLE:
            end_idx = i
            break

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        raise RuntimeError("Failed to locate section 4.2 boundaries.")

    start_p = doc.paragraphs[start_idx]._p
    end_p = doc.paragraphs[end_idx]._p
    remove_tables_between(start_p, end_p)

    for idx in range(end_idx - 1, start_idx, -1):
        delete_paragraph(doc.paragraphs[idx])

    anchor = doc.paragraphs[start_idx + 1]
    rebuild_section_4_2(doc, anchor)

    doc.save(str(doc_path))
    print(f"SAVED:{doc_path}")


if __name__ == "__main__":
    main()
