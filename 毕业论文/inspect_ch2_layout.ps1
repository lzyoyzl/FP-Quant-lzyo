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
        Write-Output ("TABLES={0}" -f $doc.Tables.Count)
        for ($i = 1; $i -le $doc.Tables.Count; $i++) {
            $tbl = $doc.Tables.Item($i)
            $startPara = $tbl.Range.Information(12)
            $endPara = $tbl.Range.Information(13)
            $rows = $tbl.Rows.Count
            $cols = $tbl.Columns.Count
            $sample = Clean-Text ($tbl.Cell(1,1).Range.Text)
            if ($sample.Length -gt 80) {
                $sample = $sample.Substring(0, 80)
            }
            Write-Output ("TABLE[{0}]=P{1}-P{2}|{3}x{4}|{5}" -f $i, $startPara, $endPara, $rows, $cols, $sample)
        }

        for ($i = 260; $i -le [Math]::Min(360, $doc.Paragraphs.Count); $i++) {
            $para = $doc.Paragraphs.Item($i)
            $text = Clean-Text $para.Range.Text
            if ([string]::IsNullOrWhiteSpace($text)) {
                continue
            }
            $inTable = $para.Range.Information(12) -ne 0
            try {
                $style = [string]$para.Range.Style.NameLocal
            } catch {
                $style = [string]$para.Range.Style
            }
            if ($text.Length -gt 120) {
                $text = $text.Substring(0, 120)
            }
            Write-Output ("P{0}|TABLE={1}|STYLE={2}|TEXT={3}" -f $i, $inTable, $style, $text)
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
