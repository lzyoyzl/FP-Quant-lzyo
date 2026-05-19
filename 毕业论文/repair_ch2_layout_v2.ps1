param(
    [Parameter(Mandatory = $true)][string]$Path
)

$ErrorActionPreference = 'Stop'

function Clean-Text($text) {
    $clean = $text -replace "[\r\n\f]+", ' '
    return $clean.Trim([char]13, [char]7, ' ', "`t")
}

function Find-LastParagraphIndex($doc, $pattern) {
    for ($i = $doc.Paragraphs.Count; $i -ge 1; $i--) {
        $text = Clean-Text $doc.Paragraphs.Item($i).Range.Text
        if ($text -match $pattern) {
            return $i
        }
    }
    throw "Paragraph not found: $pattern"
}

function Find-ExactParagraphIndex($doc, $text) {
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $current = Clean-Text $doc.Paragraphs.Item($i).Range.Text
        if ($current -eq $text) {
            return $i
        }
    }
    throw "Exact paragraph not found: $text"
}

function Set-NormalParagraphStyle($para, $alignment = 3) {
    $para.Alignment = $alignment
    $para.Range.Font.Bold = 0
    $para.Range.Font.Size = 12
    $para.Range.ParagraphFormat.SpaceAfter = 0
    $para.Range.ParagraphFormat.SpaceBefore = 0
    $para.Range.ParagraphFormat.LineSpacingRule = 0
}

function Set-CaptionStyle($para) {
    $para.Alignment = 1
    $para.Range.Font.Bold = 0
    $para.Range.Font.Size = 10.5
    $para.Range.ParagraphFormat.SpaceAfter = 0
    $para.Range.ParagraphFormat.SpaceBefore = 0
}

function Insert-TextAfterPosition($doc, [ref]$position, $text, $caption = $false) {
    $range = $doc.Range($position.Value, $position.Value)
    $range.InsertAfter($text + "`r")
    $idx = Find-LastParagraphIndex $doc ("^" + [regex]::Escape($text) + "$")
    $para = $doc.Paragraphs.Item($idx)
    if ($caption) {
        Set-CaptionStyle $para
    } else {
        Set-NormalParagraphStyle $para
    }
    $position.Value = $para.Range.End
}

function Insert-PlaceholderAfterPosition($doc, [ref]$position, $token) {
    $range = $doc.Range($position.Value, $position.Value)
    $range.InsertAfter($token + "`r")
    $idx = Find-ExactParagraphIndex $doc $token
    $para = $doc.Paragraphs.Item($idx)
    $position.Value = $para.Range.End
}

function Replace-PlaceholderWithImage($doc, $token, $imagePath, $width) {
    $idx = Find-ExactParagraphIndex $doc $token
    $para = $doc.Paragraphs.Item($idx)
    $start = $para.Range.Start
    $para.Range.Text = "`r"
    $imgRange = $doc.Range($start, $start)
    $shape = $doc.InlineShapes.AddPicture($imagePath, $false, $true, $imgRange)
    $shape.LockAspectRatio = -1
    $shape.Width = $width
    $para = $doc.Paragraphs.Item($idx)
    $para.Alignment = 1
    $para.Range.ParagraphFormat.SpaceAfter = 0
    $para.Range.ParagraphFormat.SpaceBefore = 0
}

function Format-Table($table, $headerColor) {
    $table.Borders.Enable = 1
    $table.Range.Font.Size = 10.5
    $table.Range.Font.Bold = 0
    $table.Range.ParagraphFormat.SpaceAfter = 0
    $table.Range.ParagraphFormat.SpaceBefore = 0
    $table.Range.Cells.VerticalAlignment = 1
    $table.Rows.Alignment = 1

    for ($r = 1; $r -le $table.Rows.Count; $r++) {
        for ($c = 1; $c -le $table.Columns.Count; $c++) {
            $cell = $table.Cell($r, $c)
            $cell.Range.ParagraphFormat.SpaceAfter = 0
            $cell.Range.ParagraphFormat.SpaceBefore = 0
            $cell.Range.ParagraphFormat.LineSpacingRule = 0
            if ($r -eq 1) {
                $cell.Range.Font.Bold = 1
                $cell.Range.Font.Color = 16777215
                $cell.Range.ParagraphFormat.Alignment = 1
                $cell.Shading.BackgroundPatternColor = $headerColor
            } elseif ($c -eq 1) {
                $cell.Range.ParagraphFormat.Alignment = 1
            } else {
                $cell.Range.ParagraphFormat.Alignment = 0
            }
        }
    }
}

function Replace-PlaceholderWithTable($doc, $token, $headers, $rows, $headerColor) {
    $idx = Find-ExactParagraphIndex $doc $token
    $para = $doc.Paragraphs.Item($idx)
    $start = $para.Range.Start
    $para.Range.Text = ""
    $table = $doc.Tables.Add($doc.Range($start, $start), $rows.Count + 1, $headers.Count)

    for ($c = 1; $c -le $headers.Count; $c++) {
        $table.Cell(1, $c).Range.Text = $headers[$c - 1]
    }
    for ($r = 0; $r -lt $rows.Count; $r++) {
        for ($c = 0; $c -lt $headers.Count; $c++) {
            $table.Cell($r + 2, $c + 1).Range.Text = $rows[$r][$c]
        }
    }

    Format-Table $table $headerColor
}

function Replace-RangeWithStructuredBlock($doc, $startPos, $endPos, $items) {
    $range = $doc.Range($startPos, $endPos)
    $range.Text = ""
    $position = $startPos
    foreach ($item in $items) {
        if ($item.type -eq 'text') {
            Insert-TextAfterPosition $doc ([ref]$position) $item.text $false
        } elseif ($item.type -eq 'caption') {
            Insert-TextAfterPosition $doc ([ref]$position) $item.text $true
        } elseif ($item.type -eq 'image') {
            Insert-PlaceholderAfterPosition $doc ([ref]$position) $item.token
        } elseif ($item.type -eq 'table') {
            Insert-PlaceholderAfterPosition $doc ([ref]$position) $item.token
        }
    }

    foreach ($item in $items) {
        if ($item.type -eq 'image') {
            Replace-PlaceholderWithImage $doc $item.token $item.path $item.width
        } elseif ($item.type -eq 'table') {
            Replace-PlaceholderWithTable $doc $item.token $item.headers $item.rows $item.headerColor
        }
    }
}

function Remove-ControlMarkers($doc) {
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $raw = [string]$para.Range.Text
        if ($raw.Contains([string][char]1)) {
            $para.Range.Text = $raw.Replace([string][char]1, '')
        }
    }
}

$root = Split-Path -Parent $Path
$fig21 = Join-Path $root 'generated_figures\fig2_1_transformer_block.png'
$fig22 = Join-Path $root 'generated_figures\fig2_2_rotation_quant_pipeline.png'

$table21Headers = @('线性层类别', '主要功能', '常见统计特征', '量化与旋转关注点')
$table21Rows = @(
    @('q / k / v', '构造查询、键和值表示', '与上下文建模强相关，通道重要性差异明显', '兼顾注意力分数稳定性与离群通道抑制'),
    @('o_proj', '完成多头信息融合并回到隐藏空间', '直接影响残差主干，误差易向后传播', '关注融合误差与主干稳定性'),
    @('gate / up', '进行维度扩展与非线性调制', '中间维度大，易出现重尾和局部峰值', '关注 group 共享 scale 下的峰值控制'),
    @('down_proj', '将中间表示回投影到隐藏空间', '受前序激活与扩展分支共同影响', '关注回投影误差对输出分布的扰动')
)

$table22Headers = @('比较维度', 'NVFP4', 'MXFP4', '对旋转选择的影响')
$table22Rows = @(
    @('group 粒度', '通常更细，强调局部补偿', '通常更关注块级统一表示', '决定旋转更偏局部保持还是全局扩散'),
    @('scale 使用方式', '更依赖细粒度块内稳定统计', '更依赖组内整体尺度匹配', '影响峰值扩散是否会破坏共享 scale 效率'),
    @('误差敏感点', '跨块扩散可能削弱局部隔离优势', '局部峰值可能挤压整块分辨率', '同一旋转在两种格式下未必同时最优'),
    @('部署关注点', '保持块内均衡并控制离群值', '保持统一块级尺度的利用效率', '需要结合具体格式约束进行差异化搜索')
)

$block21 = @(
    @{ type = 'image'; token = '[[FIG2_1]]'; path = $fig21; width = 405 },
    @{ type = 'caption'; text = '图2-1　Transformer block 结构与本文关注的线性层位置' },
    @{ type = 'text'; text = '表2-1 对 Transformer block 中典型线性层的功能差异与量化关注点进行了归纳。' },
    @{ type = 'caption'; text = '表2-1　Transformer block 中典型线性层的功能差异' },
    @{ type = 'table'; token = '[[TAB2_1]]'; headers = $table21Headers; rows = $table21Rows; headerColor = 5329233 }
)

$block22 = @(
    @{ type = 'text'; text = '表2-2 总结了 NVFP4 与 MXFP4 在 microscaling 机制上的主要差异，以及这些差异对旋转选择的直接影响。' },
    @{ type = 'caption'; text = '表2-2　NVFP4 与 MXFP4 的特征比较' },
    @{ type = 'table'; token = '[[TAB2_2]]'; headers = $table22Headers; rows = $table22Rows; headerColor = 5389854 }
)

$block23 = @(
    @{ type = 'image'; token = '[[FIG2_2]]'; path = $fig22; width = 405 },
    @{ type = 'caption'; text = '图2-2　低比特量化与旋转辅助的等价部署关系' }
)

$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open($Path, $false, $false)
    try {
        $pFigLead = Find-LastParagraphIndex $doc '^如图2-1所示'
        $p22 = Find-LastParagraphIndex $doc '^2\.2[　 ]'
        $pTable22Lead = Find-LastParagraphIndex $doc '^表2-2 总结了'
        $p24 = Find-LastParagraphIndex $doc '^2\.4[　 ]'
        $pFig22Lead = Find-LastParagraphIndex $doc '^图2-2 给出了'
        $p25 = Find-LastParagraphIndex $doc '^2\.5[　 ]'

        Replace-RangeWithStructuredBlock $doc $doc.Paragraphs.Item($pFig22Lead).Range.End $doc.Paragraphs.Item($p25).Range.Start $block23
        Replace-RangeWithStructuredBlock $doc $doc.Paragraphs.Item($pTable22Lead).Range.Start $doc.Paragraphs.Item($p24).Range.Start $block22
        Replace-RangeWithStructuredBlock $doc $doc.Paragraphs.Item($pFigLead).Range.End $doc.Paragraphs.Item($p22).Range.Start $block21

        Remove-ControlMarkers $doc
        $doc.Repaginate()
        if ($doc.TablesOfContents.Count -gt 0) {
            $doc.TablesOfContents.Item(1).Update()
        }
        $doc.Save()
    } finally {
        $doc.Close()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
    }
} finally {
    if ($word -ne $null) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [gc]::Collect()
    [gc]::WaitForPendingFinalizers()
}
