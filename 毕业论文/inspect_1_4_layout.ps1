$ErrorActionPreference = 'Stop'

$docPath = "\\wsl.localhost\Ubuntu-22.04\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
$doc = $word.Documents.Open($docPath, $false, $true)

try {
    $paras = $doc.Paragraphs
    $start = 220
    $end = [Math]::Min(240, $paras.Count)
    for ($i = $start; $i -le $end; $i++) {
        $para = $paras.Item($i)
        $text = $para.Range.Text.Replace("`r", "").Replace([char]7, "").Replace("`n", " ")
        $style = $para.Range.Style.NameLocal
        $outline = $para.OutlineLevel
        Write-Output ("P{0}: STYLE={1}; OL={2}; TEXT={3}" -f $i, $style, $outline, $text)
    }
}
finally {
    $doc.Close([ref]$false)
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    [gc]::Collect()
    [gc]::WaitForPendingFinalizers()
}
