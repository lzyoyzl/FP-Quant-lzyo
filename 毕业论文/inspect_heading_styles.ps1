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
        for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
            $para = $doc.Paragraphs.Item($i)
            $text = Clean-Text $para.Range.Text
            if ([string]::IsNullOrWhiteSpace($text)) {
                continue
            }
            if ($text -match '^(第[12]章[　 ].*|[12]\.[0-9]+[　 ].*|[12]\.[0-9]+\.[0-9]+[　 ].*)$') {
                try {
                    $style = [string]$para.Range.Style.NameLocal
                } catch {
                    $style = [string]$para.Range.Style
                }
                Write-Output ("P{0}|STYLE={1}|ALIGN={2}|SIZE={3}|BOLD={4}|OUTLINE={5}|TEXT={6}" -f `
                    $i,
                    $style,
                    $para.Alignment,
                    $para.Range.Font.Size,
                    $para.Range.Font.Bold,
                    $para.OutlineLevel,
                    $text)
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
