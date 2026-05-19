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
    for idx, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if "2.1" in text or "2.2" in text or 160 <= idx <= 210:
            style = para.style.name if para.style is not None else "<None>"
            print(f"P{idx}: STYLE={style}; TEXT={text}")


if __name__ == "__main__":
    main()
