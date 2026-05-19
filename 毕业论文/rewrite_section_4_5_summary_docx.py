from docx import Document


DOCX_PATH = "欧阳照林-毕业论文.docx"


NEW_TEXT = (
    "本章围绕面向低比特量化的 group-wise 旋转算法实现展开，首先在量化格式约束下明确了四类输入槽位上的局部搜索粒度与候选变换空间，"
    "随后结合当前代码实现说明了 Identity、Householder、Hadamard/GSR、DCT/DST 等候选矩阵的构造方式，以及 MSE、COV、J_tail 和 AUTO "
    "等搜索策略的评分依据与适用场景。在此基础上，进一步梳理了 group-wise 旋转搜索的任务组织、逐 group 评分、最优候选选择与 "
    "MixedGroupTransform 组装流程，形成了从候选构造到可部署变换导出的完整实现链路。由此可见，本文方法已经在候选空间、目标函数和算法流程三个层面完成工程化落地。下一章将在此基础上对整体量化效果、不同搜索策略与不同量化格式下的实验表现进行系统分析。"
)


def main():
    doc = Document(DOCX_PATH)
    paras = doc.paragraphs

    start = None
    for i, p in enumerate(paras):
        txt = p.text.strip()
        style = p.style.name if p.style else ""
        if txt.startswith("4.5 ") and style.startswith("Heading"):
            start = i
            break
    if start is None:
        raise RuntimeError("未找到 4.5 本章小结")

    next_heading = None
    for i in range(start + 1, len(paras)):
        style = paras[i].style.name if paras[i].style else ""
        txt = paras[i].text.strip()
        if style.startswith("Heading") and txt.startswith("第5章"):
            next_heading = i
            break
    if next_heading is None:
        raise RuntimeError("未找到第5章边界")

    body_idx = start + 1
    paras[body_idx].text = NEW_TEXT

    for i in range(body_idx + 1, next_heading):
        paras[i].text = ""

    doc.save(DOCX_PATH)
    print("updated 4.5 summary")


if __name__ == "__main__":
    main()
