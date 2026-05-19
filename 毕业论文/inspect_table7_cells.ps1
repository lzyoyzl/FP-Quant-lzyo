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
        $tbl = $doc.Tables.Item(7)
        Write-Output ("ROWS={0};COLS={1}" -f $tbl.Rows.Count, $tbl.Columns.Count)
        for ($r = 1; $r -le $tbl.Rows.Count; $r++) {
            for ($c = 1; $c -le $tbl.Columns.Count; $c++) {
                $txt = Clean-Text $tbl.Cell($r, $c).Range.Text
                if ($txt.Length -gt 160) {
                    $txt = $txt.Substring(0, 160)
                }
                Write-Output ("CELL[{0},{1}]={2}" -f $r, $c, $txt)
            }
        }

        for ($i = 348; $i -le [Math]::Min(430, $doc.Paragraphs.Count); $i++) {
            $para = $doc.Paragraphs.Item($i)
            $text = Clean-Text $para.Range.Text
            if ([string]::IsNullOrWhiteSpace($text)) {
                continue
            }
            $inTable = $para.Range.Information(12) -ne 0
            if ($text.Length -gt 120) {
                $text = $text.Substring(0, 120)
            }
            Write-Output ("P{0}|TABLE={1}|TEXT={2}" -f $i, $inTable, $text)
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
