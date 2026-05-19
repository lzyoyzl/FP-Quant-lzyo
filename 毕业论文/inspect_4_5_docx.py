from docx import Document

doc = Document("欧阳照林-毕业论文.docx")
paras = doc.paragraphs

for i, p in enumerate(paras):
    txt = p.text.strip()
    style = p.style.name if p.style else ""
    if txt.startswith("4.5 ") or txt.startswith("4.6 ") or txt.startswith("第5章"):
        print(f"{i}: [{style}] {txt}")

print("---- body slice ----")
start = None
for i, p in enumerate(paras):
    txt = p.text.strip()
    style = p.style.name if p.style else ""
    if txt.startswith("4.5 ") and style.startswith("Heading"):
        start = i
        break

if start is None:
    raise SystemExit("body 4.5 not found")

end = len(paras)
for i in range(start + 1, len(paras)):
    txt = paras[i].text.strip()
    style = paras[i].style.name if paras[i].style else ""
    if style.startswith("Heading") and (txt.startswith("4.6 ") or txt.startswith("第5章")):
        end = i
        break

for i in range(start, end):
    print(f"{i}: [{paras[i].style.name}] {paras[i].text}")
