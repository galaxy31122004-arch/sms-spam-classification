param(
    [string]$Path = (Join-Path $PSScriptRoot '..\drafts\bao-cao-sms-spam.docx')
)
# Word does what python-docx cannot: turn the UnicodeMath written between the two sentinels into
# real equations, fill the table of contents, and lay the document out so page numbers are real.
#
# Each formula is bookmarked before anything is edited. Word keeps a bookmark on the same text
# while the document changes, so deleting the sentinels and building one equation cannot shift
# the position of the next one.
$ErrorActionPreference = 'Stop'
$full = (Resolve-Path $Path).Path
$mathStart = [char]0xE000
$mathEnd = [char]0xE001

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($full)

    # 1. Bookmark every formula body, sentinels excluded.
    $marked = 0
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $range = $doc.Paragraphs.Item($i).Range
        $text = $range.Text
        $from = 0
        while ($true) {
            $open = $text.IndexOf($mathStart, $from)
            if ($open -lt 0) { break }
            $close = $text.IndexOf($mathEnd, $open + 1)
            if ($close -lt 0) { break }
            $marked++
            $body = $doc.Range($range.Start + $open + 1, $range.Start + $close)
            $doc.Bookmarks.Add("CongThuc$marked", $body) | Out-Null
            $from = $close + 1
        }
    }
    "Da danh dau $marked cong thuc"

    # 2. Mark each bookmarked range as an equation, then let Word build the whole collection.
    # The object returned by OMaths.Add is late-bound and has no callable BuildUp; the collection
    # method does have one, so the formulas are added first and built in one go.
    $added = 0
    $failed = @()
    for ($i = 1; $i -le $marked; $i++) {
        $name = "CongThuc$i"
        if (-not $doc.Bookmarks.Exists($name)) { $failed += "$name (mat bookmark)"; continue }
        $range = $doc.Bookmarks.Item($name).Range
        try {
            $doc.OMaths.Add($range) | Out-Null
            $added++
        }
        catch { $failed += $range.Text }
    }
    $doc.OMaths.BuildUp()
    $built = $added
    foreach ($bookmark in @($doc.Bookmarks | ForEach-Object { $_.Name })) {
        if ($bookmark -like "CongThuc*") { $doc.Bookmarks.Item($bookmark).Delete() }
    }

    # 3. Drop the sentinels, which now sit just outside each equation. Word's Find does not match
    # private-use characters, so walk the affected paragraphs backwards and delete them by hand.
    $removed = 0
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $range = $doc.Paragraphs.Item($i).Range
        if ($range.Text.IndexOf($mathStart) -lt 0 -and $range.Text.IndexOf($mathEnd) -lt 0) { continue }
        $characters = $range.Characters
        for ($c = $characters.Count; $c -ge 1; $c--) {
            $character = $characters.Item($c)
            if ($character.Text -eq $mathStart -or $character.Text -eq $mathEnd) {
                $character.Delete() | Out-Null
                $removed++
            }
        }
    }
    "Da xoa $removed ky tu moc"
    "Da dung $built / $marked bieu thuc toan"
    if ($failed.Count -gt 0) {
        "KHONG DUNG DUOC $($failed.Count):"
        $failed | Select-Object -First 8 | ForEach-Object { "   $_" }
    }

    $doc.Save()
    $doc.Close(0)

    # Fields are updated in a second pass on a freshly opened document: after the equation edits
    # the first handle no longer exposes the Fields collection reliably.
    $doc = $word.Documents.Open($full)
    $doc.Fields.Update() | Out-Null
    for ($i = 1; $i -le $doc.TablesOfContents.Count; $i++) {
        $doc.TablesOfContents.Item($i).Update()
    }
    $doc.Repaginate()
    "So trang sau khi dan trang: $($doc.ComputeStatistics(2))"
    "So tu: $($doc.ComputeStatistics(0))"
    "So bieu thuc trong tai lieu: $($doc.OMaths.Count)"
    $doc.Save()
    $doc.Close(0)
}
finally {
    $word.Quit()
    [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
