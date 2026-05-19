param(
    [Parameter(Mandatory = $true)][string]$Path
)

$ErrorActionPreference = 'Stop'

function Get-NormalizedCellText($cell) {
    $text = $cell.Range.Text
    return ($text -replace "[\r\n\f]+", ' ').Trim([char]13, [char]7, ' ', "`t")
}

$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open($Path, $false, $false)
    try {
        $targetIndex = 0
        for ($i = 1; $i -le $doc.Tables.Count; $i++) {
            $tbl = $doc.Tables.Item($i)
            $sample = Get-NormalizedCellText $tbl.Cell(1,1)
            if ($tbl.Rows.Count -eq 3 -and $tbl.Columns.Count -eq 4 -and $sample.Length -gt 500) {
                $targetIndex = $i
                break
            }
        }

        if ($targetIndex -eq 0) {
            throw "Abnormal chapter table not found."
        }

        $target = $doc.Tables.Item($targetIndex)
        $start = $target.Range.Start
        $raw = $target.Cell(1,1).Range.Text.TrimEnd([char]13, [char]7)
        $target.Delete()
        $doc.Range($start, $start).InsertBefore($raw + "`r")
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

