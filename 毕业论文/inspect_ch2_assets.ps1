param(
    [Parameter(Mandatory = $true)][string]$Path
)

$ErrorActionPreference = 'Stop'

function Clean-Text($text) {
    $clean = $text -replace "[\r\n\f]+", ' '
    return $clean.Trim([char]13, [char]7, ' ', "`t")
}

$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open($Path, $false, $true)
    try {
        Write-Output ("PARAGRAPHS={0}" -f $doc.Paragraphs.Count)
        Write-Output ("TABLES={0}" -f $doc.Tables.Count)
        Write-Output ("INLINESHAPES={0}" -f $doc.InlineShapes.Count)
        Write-Output ("SHAPES={0}" -f $doc.Shapes.Count)

        for ($i = 1; $i -le $doc.Tables.Count; $i++) {
            $tbl = $doc.Tables.Item($i)
            $sample = Clean-Text ($tbl.Cell(1,1).Range.Text)
            if ($sample.Length -gt 60) {
                $sample = $sample.Substring(0, 60)
            }
            Write-Output ("TABLE[{0}]={1}x{2}|{3}" -f $i, $tbl.Rows.Count, $tbl.Columns.Count, $sample)
        }

        for ($i = 296; $i -le [Math]::Min(390, $doc.Paragraphs.Count); $i++) {
            $para = $doc.Paragraphs.Item($i)
            $txt = Clean-Text $para.Range.Text
            if ([string]::IsNullOrWhiteSpace($txt)) {
                continue
            }
            $inTable = $para.Range.Information(12) -ne 0
            if ($txt.Length -gt 120) {
                $txt = $txt.Substring(0, 120)
            }
            Write-Output ("P{0}|TABLE={1}|TEXT={2}" -f $i, $inTable, $txt)
        }

        foreach ($idx in @(5, 6, 7)) {
            if ($idx -le $doc.Tables.Count) {
                $tbl = $doc.Tables.Item($idx)
                Write-Output ("TABLEDETAIL[{0}]={1}x{2}" -f $idx, $tbl.Rows.Count, $tbl.Columns.Count)
                for ($r = 1; $r -le $tbl.Rows.Count; $r++) {
                    for ($c = 1; $c -le $tbl.Columns.Count; $c++) {
                        $txt = Clean-Text $tbl.Cell($r, $c).Range.Text
                        if ($txt.Length -gt 100) {
                            $txt = $txt.Substring(0, 100)
                        }
                        Write-Output ("T{0}[{1},{2}]={3}" -f $idx, $r, $c, $txt)
                    }
                }
            }
        }
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
