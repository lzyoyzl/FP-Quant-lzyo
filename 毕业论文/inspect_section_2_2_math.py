from pathlib import Path
import sys

from docx import Document


DOC_PATH = Path(r"\\wsl.localhost\Ubuntu-22.04\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    doc = Document(str(DOC_PATH))
    for idx in range(193, 207):
        para = doc.paragraphs[idx]
        style = para.style.name if para.style is not None else "<None>"
        math_count = len(para._p.xpath(".//*[local-name()='oMath']"))
        math_text = "".join(t.text or "" for t in para._p.xpath(".//*[local-name()='oMath']//*[local-name()='t']"))
        print(f"P{idx}: STYLE={style}; MATH={math_count}; MATH_TEXT={math_text}")


if __name__ == "__main__":
    main()
