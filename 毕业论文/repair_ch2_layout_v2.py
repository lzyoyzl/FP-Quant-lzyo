from __future__ import annotations

import re
import sys
from pathlib import Path

import pythoncom
import win32com.client


def clean_text(text: str) -> str:
    return re.sub(r"[\r\n\f]+", " ", text).strip("\r\x07 \t")


def find_last_paragraph_index(doc, pattern: str) -> int:
    rgx = re.compile(pattern)
    for i in range(doc.Paragraphs.Count, 0, -1):
        text = clean_text(doc.Paragraphs(i).Range.Text)
        if rgx.search(text):
            return i
    raise RuntimeError(f"Paragraph not found: {pattern}")


def find_exact_paragraph_index(doc, target: str) -> int:
    for i in range(1, doc.Paragraphs.Count + 1):
        text = clean_text(doc.Paragraphs(i).Range.Text)
        if text == target:
            return i
    raise RuntimeError(f"Exact paragraph not found: {target}")


def set_normal_paragraph_style(para) -> None:
    para.Alignment = 3
    para.Range.Font.Bold = 0
    para.Range.Font.Size = 12
    para.Range.ParagraphFormat.SpaceAfter = 0
    para.Range.ParagraphFormat.SpaceBefore = 0


def set_caption_style(para) -> None:
    para.Alignment = 1
    para.Range.Font.Bold = 0
    para.Range.Font.Size = 10.5
    para.Range.ParagraphFormat.SpaceAfter = 0
    para.Range.ParagraphFormat.SpaceBefore = 0


def insert_text_after_position(doc, position: int, text: str, *, caption: bool) -> int:
    rng = doc.Range(position, position)
    rng.InsertAfter(text + "\r")
    idx = find_exact_paragraph_index(doc, text)
    para = doc.Paragraphs(idx)
    if caption:
        set_caption_style(para)
    else:
        set_normal_paragraph_style(para)
    return para.Range.End


def insert_placeholder_after_position(doc, position: int, token: str) -> int:
    rng = doc.Range(position, position)
    rng.InsertAfter(token + "\r")
    idx = find_exact_paragraph_index(doc, token)
    return doc.Paragraphs(idx).Range.End


def replace_placeholder_with_image(doc, token: str, image_path: Path, width: int) -> None:
    idx = find_exact_paragraph_index(doc, token)
    para = doc.Paragraphs(idx)
    start = para.Range.Start
    para.Range.Text = "\r"
    img_range = doc.Range(start, start)
    shape = doc.InlineShapes.AddPicture(str(image_path), False, True, img_range)
    shape.LockAspectRatio = -1
    shape.Width = width
    para = doc.Paragraphs(idx)
    para.Alignment = 1
    para.Range.ParagraphFormat.SpaceAfter = 0
    para.Range.ParagraphFormat.SpaceBefore = 0


def format_table(table, header_color: int) -> None:
    table.Borders.Enable = 1
    table.Range.Font.Size = 10.5
    table.Range.Font.Bold = 0
    table.Range.ParagraphFormat.SpaceAfter = 0
    table.Range.ParagraphFormat.SpaceBefore = 0
    table.Rows.Alignment = 1
    table.Range.Cells.VerticalAlignment = 1

    for r in range(1, table.Rows.Count + 1):
        for c in range(1, table.Columns.Count + 1):
            cell = table.Cell(r, c)
            cell.Range.ParagraphFormat.SpaceAfter = 0
            cell.Range.ParagraphFormat.SpaceBefore = 0
            if r == 1:
                cell.Range.Font.Bold = 1
                cell.Range.Font.Color = 16777215
                cell.Range.ParagraphFormat.Alignment = 1
                cell.Shading.BackgroundPatternColor = header_color
            elif c == 1:
                cell.Range.ParagraphFormat.Alignment = 1
            else:
                cell.Range.ParagraphFormat.Alignment = 0


def replace_placeholder_with_table(doc, token: str, headers, rows, header_color: int) -> None:
    idx = find_exact_paragraph_index(doc, token)
    para = doc.Paragraphs(idx)
    start = para.Range.Start
    para.Range.Text = ""
    table = doc.Tables.Add(doc.Range(start, start), len(rows) + 1, len(headers))

    for c, header in enumerate(headers, start=1):
        table.Cell(1, c).Range.Text = header

    for r, row in enumerate(rows, start=2):
        for c, value in enumerate(row, start=1):
            table.Cell(r, c).Range.Text = value

    format_table(table, header_color)


def replace_range_with_structured_block(doc, start_pos: int, end_pos: int, items) -> None:
    doc.Range(start_pos, end_pos).Text = ""
    position = start_pos
    for item in items:
        if item["type"] == "text":
            position = insert_text_after_position(doc, position, item["text"], caption=False)
        elif item["type"] == "caption":
            position = insert_text_after_position(doc, position, item["text"], caption=True)
        elif item["type"] in {"image", "table"}:
            position = insert_placeholder_after_position(doc, position, item["token"])

    for item in items:
        if item["type"] == "image":
            replace_placeholder_with_image(doc, item["token"], item["path"], item["width"])
        elif item["type"] == "table":
            replace_placeholder_with_table(
                doc,
                item["token"],
                item["headers"],
                item["rows"],
                item["header_color"],
            )


def remove_control_markers(doc) -> None:
    for i in range(1, doc.Paragraphs.Count + 1):
        para = doc.Paragraphs(i)
        raw = str(para.Range.Text)
        if "\x01" in raw:
            para.Range.Text = raw.replace("\x01", "")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python repair_ch2_layout_v2.py <doc-path>")
        return 2

    doc_path = Path(sys.argv[1]).resolve()
    root = doc_path.parent
    fig21 = root / "generated_figures" / "fig2_1_transformer_block.png"
    fig22 = root / "generated_figures" / "fig2_2_rotation_quant_pipeline.png"

    table21_headers = ["线性层类别", "主要功能", "常见统计特征", "量化与旋转关注点"]
    table21_rows = [
        ["q / k / v", "构造查询、键和值表示", "与上下文建模强相关，通道重要性差异明显", "兼顾注意力分数稳定性与离群通道抑制"],
        ["o_proj", "完成多头信息融合并回到隐藏空间", "直接影响残差主干，误差易向后传播", "关注融合误差与主干稳定性"],
        ["gate / up", "进行维度扩展与非线性调制", "中间维度大，易出现重尾和局部峰值", "关注 group 共享 scale 下的峰值控制"],
        ["down_proj", "将中间表示回投影到隐藏空间", "受前序激活与扩展分支共同影响", "关注回投影误差对输出分布的扰动"],
    ]

    table22_headers = ["比较维度", "NVFP4", "MXFP4", "对旋转选择的影响"]
    table22_rows = [
        ["group 粒度", "通常更细，强调局部补偿", "通常更关注块级统一表示", "决定旋转更偏局部保持还是全局扩散"],
        ["scale 使用方式", "更依赖细粒度块内稳定统计", "更依赖组内整体尺度匹配", "影响峰值扩散是否会破坏共享 scale 效率"],
        ["误差敏感点", "跨块扩散可能削弱局部隔离优势", "局部峰值可能挤压整块分辨率", "同一旋转在两种格式下未必同时最优"],
        ["部署关注点", "保持块内均衡并控制离群值", "保持统一块级尺度的利用效率", "需要结合具体格式约束进行差异化搜索"],
    ]

    block21 = [
        {"type": "image", "token": "[[FIG2_1]]", "path": fig21, "width": 405},
        {"type": "caption", "text": "图2-1　Transformer block 结构与本文关注的线性层位置"},
        {"type": "text", "text": "表2-1 对 Transformer block 中典型线性层的功能差异与量化关注点进行了归纳。"},
        {"type": "caption", "text": "表2-1　Transformer block 中典型线性层的功能差异"},
        {
            "type": "table",
            "token": "[[TAB2_1]]",
            "headers": table21_headers,
            "rows": table21_rows,
            "header_color": 5329233,
        },
    ]

    block22 = [
        {"type": "text", "text": "表2-2 总结了 NVFP4 与 MXFP4 在 microscaling 机制上的主要差异，以及这些差异对旋转选择的直接影响。"},
        {"type": "caption", "text": "表2-2　NVFP4 与 MXFP4 的特征比较"},
        {
            "type": "table",
            "token": "[[TAB2_2]]",
            "headers": table22_headers,
            "rows": table22_rows,
            "header_color": 5389854,
        },
    ]

    block23 = [
        {"type": "image", "token": "[[FIG2_2]]", "path": fig22, "width": 405},
        {"type": "caption", "text": "图2-2　低比特量化与旋转辅助的等价部署关系"},
    ]

    pythoncom.CoInitialize()
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0

    try:
        doc = word.Documents.Open(str(doc_path), False, False)
        try:
            p_fig1 = find_last_paragraph_index(doc, r"^如图2-1所示")
            p_22 = find_last_paragraph_index(doc, r"^2\.2[　 ]")
            p_table22 = find_last_paragraph_index(doc, r"^表2-2 总结了")
            p_24 = find_last_paragraph_index(doc, r"^2\.4[　 ]")
            p_fig2 = find_last_paragraph_index(doc, r"^图2-2 给出了")
            p_25 = find_last_paragraph_index(doc, r"^2\.5[　 ]")

            replace_range_with_structured_block(
                doc,
                doc.Paragraphs(p_fig2).Range.End,
                doc.Paragraphs(p_25).Range.Start,
                block23,
            )
            replace_range_with_structured_block(
                doc,
                doc.Paragraphs(p_table22).Range.Start,
                doc.Paragraphs(p_24).Range.Start,
                block22,
            )
            replace_range_with_structured_block(
                doc,
                doc.Paragraphs(p_fig1).Range.End,
                doc.Paragraphs(p_22).Range.Start,
                block21,
            )

            remove_control_markers(doc)
            doc.Repaginate()
            if doc.TablesOfContents.Count > 0:
                doc.TablesOfContents(1).Update()
            doc.Save()
        finally:
            doc.Close()
    finally:
        word.Quit()
        pythoncom.CoUninitialize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
