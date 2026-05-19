from pathlib import Path
import sys

from docx import Document
from docx.oxml.ns import qn


DOC_PATH = Path(r"\\wsl.localhost\Ubuntu-22.04\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx")
SECTION_TITLE = "2.1　后训练低比特量化基础"
NEXT_SECTION_TITLE = "2.2　NVFP4 与 MXFP4 微缩放量化格式"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    doc = Document(str(DOC_PATH))
    body = doc._element.body

    in_section = False
    para_idx = -1
    tbl_idx = -1

    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            para_idx += 1
            text = "".join(t.text or "" for t in child.iter(qn("w:t"))).strip()
            if text == SECTION_TITLE:
                in_section = True
                print(f"P{para_idx}: {text}")
                continue
            if in_section and text == NEXT_SECTION_TITLE:
                print(f"P{para_idx}: {text}")
                break
            if in_section:
                print(f"P{para_idx}: {text}")
        elif child.tag == qn("w:tbl"):
            tbl_idx += 1
            if in_section:
                print(f"T{tbl_idx}: <table>")


if __name__ == "__main__":
    main()
