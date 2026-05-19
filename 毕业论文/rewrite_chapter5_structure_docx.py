from docx import Document


DOCX_PATH = "欧阳照林-毕业论文.docx"


def set_para(paragraph, text, style_name=None):
    paragraph.text = text
    if style_name is not None:
        paragraph.style = style_name


def main():
    doc = Document(DOCX_PATH)
    paras = doc.paragraphs

    start = None
    for i, p in enumerate(paras):
        txt = p.text.strip()
        style = p.style.name if p.style else ""
        if txt.startswith("第5章") and style.startswith("Heading"):
            start = i
            break
    if start is None:
        raise RuntimeError("未找到第5章标题")

    # Current chapter-5 heading block is already present and mostly empty.
    # Only restructure the headings inside chapter 5.
    set_para(paras[start + 1], "5.1　实验环境与设置", "Heading 2")
    set_para(paras[start + 2], "5.2　group-wise 旋转算法整体效果分析（精度对比）", "Heading 2")
    set_para(paras[start + 3], "5.3　旋转选择结果分析", "Heading 2")
    set_para(paras[start + 4], "5.4　本章小结", "Heading 2")

    # Clear obsolete headings from the previous outline.
    for idx in range(start + 5, min(start + 10, len(paras))):
        paras[idx].text = ""
        paras[idx].style = "Normal"

    doc.save(DOCX_PATH)
    print("updated chapter 5 structure")


if __name__ == "__main__":
    main()
