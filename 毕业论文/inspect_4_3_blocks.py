import sys
from docx import Document
from docx.oxml.ns import qn

DOCX = r"Z:\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx"
START = "4.3 group-wise旋转搜索目标函数设计"
END = "4.4 group-wise旋转搜索算法流程"


def para_text(el):
    texts = el.xpath(".//w:t")
    return "".join(t.text for t in texts if t.text)


def safe_print(text):
    sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


doc = Document(DOCX)
body = doc._element.body
inside = False
idx = 0
for child in body.iterchildren():
    tag = child.tag.split("}")[-1]
    if tag == "p":
        text = para_text(child).strip()
        if text == START:
            inside = True
        if inside:
            safe_print(f"{idx}: [P] {text}")
            idx += 1
        if text == END:
            break
    elif tag == "tbl" and inside:
        safe_print(f"{idx}: [TBL]")
        idx += 1
