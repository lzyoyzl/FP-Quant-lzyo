import win32com.client as win32
from pathlib import Path


DOC_PATH = Path(r"\\?\UNC\wsl.localhost\Ubuntu-22.04\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx")


def main() -> None:
    word = win32.gencache.EnsureDispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = word.Documents.Open(str(DOC_PATH), ReadOnly=True)
    try:
        paras = doc.Paragraphs
        start = max(1, 220)
        end = min(paras.Count, 240)
        for i in range(start, end + 1):
            para = paras(i)
            text = para.Range.Text.replace("\r", "").replace("\x07", "").replace("\n", " ")
            try:
                style = para.Range.Style.NameLocal
            except Exception:
                style = str(para.Range.Style)
            try:
                outline = para.OutlineLevel
            except Exception:
                outline = "NA"
            print(f"P{i}: STYLE={style}; OL={outline}; TEXT={text}")
    finally:
        doc.Close(False)
        word.Quit()


if __name__ == "__main__":
    main()
