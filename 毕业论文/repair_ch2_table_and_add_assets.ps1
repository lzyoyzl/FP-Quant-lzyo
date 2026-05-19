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

function Insert-ParagraphBeforeTarget($doc, $targetPattern, $text, $alignment = 0) {
    $idx = Find-LastParagraphIndex $doc $targetPattern
    $range = $doc.Range($doc.Paragraphs.Item($idx).Range.Start, $doc.Paragraphs.Item($idx).Range.Start)
    $range.InsertBefore($text + "`r")
    $doc.Paragraphs.Item($idx).Alignment = $alignment
}

function Insert-ImageBeforeTarget($doc, $targetPattern, $imagePath, $width) {
    $idx = Find-LastParagraphIndex $doc $targetPattern
    $range = $doc.Range($doc.Paragraphs.Item($idx).Range.Start, $doc.Paragraphs.Item($idx).Range.Start)
    $para = $doc.Paragraphs.Add($range)
    $para.Range.Text = "`r"
    $shape = $doc.InlineShapes.AddPicture($imagePath, $false, $true, $para.Range)
    $shape.LockAspectRatio = -1
    $shape.Width = $width
    $para.Alignment = 1
}

function Insert-TableBeforeTarget($doc, $targetPattern, $headers, $rows) {
    $idx = Find-LastParagraphIndex $doc $targetPattern
    $range = $doc.Range($doc.Paragraphs.Item($idx).Range.Start, $doc.Paragraphs.Item($idx).Range.Start)
    $table = $doc.Tables.Add($range, $rows.Count + 1, $headers.Count)

    for ($c = 1; $c -le $headers.Count; $c++) {
        $table.Cell(1, $c).Range.Text = $headers[$c - 1]
        $table.Cell(1, $c).Range.Font.Bold = 1
        $table.Cell(1, $c).Range.Font.Color = 16777215
        $table.Cell(1, $c).Range.ParagraphFormat.Alignment = 1
        $table.Cell(1, $c).Shading.BackgroundPatternColor = 3355443
    }

    for ($r = 0; $r -lt $rows.Count; $r++) {
        for ($c = 0; $c -lt $headers.Count; $c++) {
            $cell = $table.Cell($r + 2, $c + 1)
            $cell.Range.Text = $rows[$r][$c]
            if ($c -eq 0) {
                $cell.Range.ParagraphFormat.Alignment = 1
            } else {
                $cell.Range.ParagraphFormat.Alignment = 0
            }
        }
    }

    $table.Range.Font.Size = 10.5
    $table.Range.ParagraphFormat.SpaceAfter = 0
    $table.Rows.Alignment = 1
    $table.Borders.Enable = 1
}

function Fix-AbnormalChapterTable($doc) {
    for ($i = 1; $i -le $doc.Tables.Count; $i++) {
        $tbl = $doc.Tables.Item($i)
        $cellText = Clean-Text $tbl.Cell(1,1).Range.Text
        if ($cellText -match '^第2章') {
            $raw = $tbl.Cell(1,1).Range.Text.TrimEnd([char]13, [char]7)
            $tbl.Range.Text = $raw + "`r"
            return
        }
    }
}

function Set-OutlineLevels($doc) {
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        $text = Clean-Text $para.Range.Text
        if ([string]::IsNullOrWhiteSpace($text)) {
            continue
        }

        $level = 10
        if ($text -match '^(摘　要|ABSTRACT|第[0-9]+章[　 ].*|致[　 ]?谢|参考文献|附　录)$') {
            $level = 1
        } elseif ($text -match '^附录[A-Z]') {
            $level = 2
        } elseif ($text -match '^[0-9]+\.[0-9]+\.[0-9]+[　 ]') {
            $level = 3
        } elseif ($text -match '^[0-9]+\.[0-9]+[　 ]') {
            $level = 2
        }
        $para.OutlineLevel = $level

        if ($text -match '^(Q =|Attention\(|x\^q =|x_tilde =|x\^q_\(g,i\) =|q_\(g,i\) =|y = xW)') {
            $para.Alignment = 1
        }
    }
}

$root = Split-Path -Parent $Path
$fig21 = Join-Path $root 'generated_figures\fig2_1_transformer_block.png'
$fig22 = Join-Path $root 'generated_figures\fig2_2_rotation_quant_pipeline.png'

$table21Headers = @('线性层类别', '主要功能', '常见统计特征', '量化与旋转关注点')
$table21Rows = @(
    @('q / k / v', '构造查询、键和值表示', '与上下文建模强相关，通道重要性差异明显', '需要兼顾注意力分数稳定性与离群通道抑制'),
    @('o_proj', '完成多头信息融合并回到隐藏空间', '直接影响残差主干，误差易向后传播', '更关注融合误差和主干稳定性'),
    @('gate / up', '进行维度扩展与非线性调制', '中间维度大，易出现重尾和局部峰值', '更关注 group 共享 scale 下的峰值控制'),
    @('down_proj', '将中间表示回投影到隐藏空间', '受前序激活与扩展分支共同影响', '更关注回投影误差对输出分布的扰动')
)

$table22Headers = @('比较维度', 'NVFP4', 'MXFP4', '对旋转选择的影响')
$table22Rows = @(
    @('group 粒度', '通常更细，强调局部补偿', '通常更关注块级统一表示', '决定旋转是更偏局部保持还是更偏全局扩散'),
    @('scale 使用方式', '更依赖细粒度块内稳定统计', '更依赖组内整体尺度匹配', '影响峰值扩散是否会破坏共享 scale 效率'),
    @('误差敏感点', '跨块扩散可能削弱局部隔离优势', '局部峰值可能挤压整块分辨率', '同一旋转在两种格式下未必同时最优'),
    @('部署关注点', '保持块内均衡并控制离群值', '保持统一块级尺度的利用效率', '需要结合具体格式约束进行差异化搜索')
)

$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open($Path, $false, $false)
    try {
        Fix-AbnormalChapterTable $doc

        Insert-ParagraphBeforeTarget $doc '^2\.2[　 ]' '如图2-1所示，decoder-only Transformer block 中的 q/k/v、o_proj 以及 gate_proj、up_proj、down_proj 等线性层共同构成了本文后续量化与旋转优化的核心对象。'
        Insert-ImageBeforeTarget $doc '^2\.2[　 ]' $fig21 405
        Insert-ParagraphBeforeTarget $doc '^2\.2[　 ]' '图2-1　Transformer block 结构与本文关注的线性层位置' 1

        Insert-ParagraphBeforeTarget $doc '^2\.2[　 ]' '表2-1 对 Transformer block 中典型线性层的功能差异与量化关注点进行了归纳。'
        Insert-ParagraphBeforeTarget $doc '^2\.2[　 ]' '表2-1　Transformer block 中典型线性层的功能差异' 1
        Insert-TableBeforeTarget $doc '^2\.2[　 ]' $table21Headers $table21Rows

        Insert-ParagraphBeforeTarget $doc '^2\.4[　 ]' '表2-2 总结了 NVFP4 与 MXFP4 在 microscaling 机制上的主要差异，以及这些差异对旋转选择的直接影响。'
        Insert-ParagraphBeforeTarget $doc '^2\.4[　 ]' '表2-2　NVFP4 与 MXFP4 的特征比较' 1
        Insert-TableBeforeTarget $doc '^2\.4[　 ]' $table22Headers $table22Rows

        Insert-ParagraphBeforeTarget $doc '^2\.5[　 ]' '图2-2 给出了低比特量化、正交旋转与等价部署之间的关系示意，体现了本文后续“激活侧前向旋转、权重侧逆变换折叠”的基本实现思路。'
        Insert-ImageBeforeTarget $doc '^2\.5[　 ]' $fig22 405
        Insert-ParagraphBeforeTarget $doc '^2\.5[　 ]' '图2-2　低比特量化与旋转辅助的等价部署关系' 1

        Set-OutlineLevels $doc
        $doc.Repaginate()
        if ($doc.TablesOfContents.Count -gt 0) {
            $doc.TablesOfContents.Item(1).Update()
        }
        foreach ($field in $doc.Fields) {
            try {
                [void]$field.Update()
            } catch {
            }
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

