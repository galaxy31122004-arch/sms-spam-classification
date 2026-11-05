param(
    [string]$Path = (Join-Path $PSScriptRoot '..\drafts\bao-cao-sms-spam.docx')
)
# Reads the paginated document through Word and reports layout facts python-docx cannot see:
# which page each figure and table lands on, whether a caption is separated from its object,
# and how much of the last line of each page is used.
$ErrorActionPreference = 'Stop'
$full = (Resolve-Path $Path).Path
$wdActiveEndPageNumber = 3

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($full, $false, $true)   # read-only
    $doc.Repaginate()
    $pages = $doc.ComputeStatistics(2)
    "So trang: $pages"

    $captionPages = @{}
    $paragraphs = $doc.Paragraphs
    for ($i = 1; $i -le $paragraphs.Count; $i++) {
        $text = $paragraphs.Item($i).Range.Text.Trim()
        if ($text -match '^(Hình|Bảng)\s+([\d.]+)\.') {
            $key = "$($Matches[1]) $($Matches[2])"
            if (-not $captionPages.ContainsKey($key)) {
                $captionPages[$key] = $paragraphs.Item($i).Range.Information($wdActiveEndPageNumber)
            }
        }
    }

    "`nHinh: trang cua anh va cua chu thich"
    $shapes = $doc.InlineShapes
    for ($i = 1; $i -le $shapes.Count; $i++) {
        $range = $shapes.Item($i).Range
        $page = $range.Information($wdActiveEndPageNumber)
        "  anh #$i -> trang $page"
    }
    foreach ($key in ($captionPages.Keys | Sort-Object)) {
        "  chu thich '$key' -> trang $($captionPages[$key])"
    }

    "`nBang: trang bat dau va ket thuc"
    for ($i = 1; $i -le $doc.Tables.Count; $i++) {
        $table = $doc.Tables.Item($i)
        $start = $table.Range.Start
        $end = $table.Range.End
        $firstPage = $doc.Range($start, $start).Information($wdActiveEndPageNumber)
        $lastPage = $doc.Range($end - 1, $end - 1).Information($wdActiveEndPageNumber)
        $flag = if ($firstPage -ne $lastPage) { "  (tach qua $($lastPage - $firstPage) trang)" } else { "" }
        "  bang $i : trang $firstPage -> $lastPage$flag"
    }

    $doc.Close(0)
}
finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
