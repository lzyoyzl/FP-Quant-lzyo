$ErrorActionPreference = 'Stop'

$docPath = "Z:\root\graduation_work\FP-Quant-lzyo\毕业论文\欧阳照林-毕业论文.docx"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

try {
    $doc = $word.Documents.OpenNoRepairDialog($docPath)
    try {
        Write-Output ("OPENED:{0}" -f $doc.Paragraphs.Count)
    }
    finally {
        $doc.Close([ref]$false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
    }
}
finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    [gc]::Collect()
    [gc]::WaitForPendingFinalizers()
}
