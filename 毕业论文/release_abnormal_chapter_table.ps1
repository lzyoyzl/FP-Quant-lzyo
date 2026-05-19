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

    $doc = $word.Documents.Open($Path, $false, $false)
    try {
        for ($i = 1; $i -le $doc.Tables.Count; $i++) {
            $tbl = $doc.Tables.Item($i)
            $firstCell = Clean-Text $tbl.Cell(1,1).Range.Text
            if ($firstCell -match '^第2章') {
                $raw = $tbl.Cell(1,1).Range.Text.TrimEnd([char]13, [char]7)
                $tbl.Range.Text = $raw + "`r"
                break
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

