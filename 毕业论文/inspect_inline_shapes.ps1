param(
    [Parameter(Mandatory = $true)][string]$Path
)

$ErrorActionPreference = 'Stop'

function Clean-Text($text) {
    $clean = $text -replace "[\r\n\f]+", ' '
    return $clean.Trim([char]13, [char]7, ' ', "`t")
}

function Get-ParagraphIndexForRange($doc, $startPos) {
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $para = $doc.Paragraphs.Item($i)
        if ($para.Range.Start -le $startPos -and $startPos -lt $para.Range.End) {
            return $i
        }
    }
    return -1
}

$word = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0

    $doc = $word.Documents.Open($Path, $false, $true)
    try {
        Write-Output ("INLINESHAPES={0}" -f $doc.InlineShapes.Count)
        for ($i = 1; $i -le $doc.InlineShapes.Count; $i++) {
            $shape = $doc.InlineShapes.Item($i)
            $pidx = Get-ParagraphIndexForRange $doc $shape.Range.Start
            $ptext = ''
            if ($pidx -gt 0) {
                $ptext = Clean-Text $doc.Paragraphs.Item($pidx).Range.Text
            }
            Write-Output ("ISHAPE[{0}]|P{1}|W={2}|H={3}|TEXT={4}" -f $i, $pidx, [int]$shape.Width, [int]$shape.Height, $ptext)
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
