from pathlib import Path

from docx import Document


DOC_PATH = Path(r"\\wsl.localhost\Ubuntu-22.04\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx")
def main() -> None:
    doc = Document(str(DOC_PATH))
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if "1.4" in text or "第2章 相关理论与技术基础" in text or 150 <= idx <= 175:
            style = para.style.name if para.style is not None else "<None>"
            has_drawing = bool(para._p.xpath(".//*[local-name()='drawing']"))
            print(f"P{idx}: STYLE={style}; DRAWING={has_drawing}; TEXT={text}")


if __name__ == "__main__":
    main()
