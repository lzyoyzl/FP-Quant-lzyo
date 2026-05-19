import sys

from docx import Document


DEFAULT_DOC_PATH = r"Z:\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx"


def main():
    doc_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DOC_PATH
    doc = Document(doc_path)
    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip().replace("\t", " ")
        style = p.style.name if p.style is not None else ""
        if 340 <= i <= 410:
            line = f"{i}: [{style}] {text}\n"
            sys.stdout.buffer.write(line.encode("utf-8", "ignore"))


if __name__ == "__main__":
    main()
