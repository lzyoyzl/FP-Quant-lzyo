from docx import Document


doc = Document("欧阳照林-毕业论文.docx")
paras = doc.paragraphs

start = None
for i, p in enumerate(paras):
    txt = p.text.strip()
    style = p.style.name if p.style else ""
    if txt.startswith("第5章") and style.startswith("Heading"):
        start = i
        break

if start is None:
    raise SystemExit("chapter 5 not found")

end = len(paras)
for i in range(start + 1, len(paras)):
    txt = paras[i].text.strip()
    style = paras[i].style.name if paras[i].style else ""
    if txt.startswith("第6章") and style.startswith("Heading"):
        end = i
        break

print(f"chapter5_start={start}, chapter5_end={end}")
for i in range(start, end):
    print(f"{i}: [{paras[i].style.name}] {paras[i].text}")
