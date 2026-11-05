param([string]$AuditDirectory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$reportRoot = (Resolve-Path (Join-Path $AuditDirectory '..')).Path
$files = @('01-mo-dau.md','02-chuong-1-co-so-ly-thuyet.md','03-chuong-2-du-lieu-va-tien-xu-ly.md','04-chuong-3-mo-hinh-va-huan-luyen.md','05-chuong-4-ket-qua-va-so-sanh.md','06-chuong-5-website-demo.md','07-ket-luan.md','08-tai-lieu-tham-khao.md','09-phu-luc.md')
$anchors = @('mo-dau','chuong-1','chuong-2','chuong-3','chuong-4','chuong-5','ket-luan','tai-lieu-tham-khao','phu-luc')
$parts = @($files | ForEach-Object { [System.IO.File]::ReadAllText((Join-Path $reportRoot $_)).TrimEnd().Replace("`r`n","`n") })
$captions = @()
foreach ($part in $parts) { $captions += @([regex]::Matches($part,'(?m)^\*\*((?:Bảng|Hình) [^\r\n]+)\*\*$') | ForEach-Object { $_.Groups[1].Value }) }
$header = "# Phát hiện tin nhắn SMS rác bằng mạng nơ-ron MLP, LSTM và Transformer`n`n## Mục lục nội dung`n`n"
for ($i=0; $i -lt $parts.Count; $i++) { $label=($parts[$i] -split "`n")[0] -replace '^# ',''; $header += "- [$label](#$($anchors[$i]))`n" }
$header += "`n## Danh mục hình`n`n"
foreach ($caption in @($captions | Where-Object { $_ -like 'Hình *' })) { $header += "- $caption`n" }
$header += "`n## Danh mục bảng`n`n"
foreach ($caption in @($captions | Where-Object { $_ -like 'Bảng *' })) { $header += "- $caption`n" }
$combined = $header.TrimEnd()
for ($i=0; $i -lt $parts.Count; $i++) { $combined += "`n`n---`n`n<a id=`"$($anchors[$i])`"></a>`n`n$($parts[$i])" }
$combined += "`n"
$combinedPath = Join-Path $reportRoot 'BAO-CAO-TONG-HOP.md'
[System.IO.File]::WriteAllText($combinedPath, $combined, [System.Text.UTF8Encoding]::new($false))
$readback = [System.IO.File]::ReadAllText($combinedPath)
if ($readback -cne $combined) { throw 'Combined output differs from its source chapters' }
for ($i=0; $i -lt $parts.Count; $i++) { if (-not $readback.Contains($parts[$i])) { throw "Missing chapter $($files[$i])" } }
$body = ($parts[0..6] + $parts[8]) -join "`n"
$used = @([regex]::Matches($body,'\[(\d{1,2})\]') | ForEach-Object { [int]$_.Groups[1].Value } | Sort-Object -Unique)
$defined = @([regex]::Matches($parts[7],'(?m)^\[(\d{1,2})\] ') | ForEach-Object { [int]$_.Groups[1].Value } | Sort-Object -Unique)
if (($used -join ',') -ne ($defined -join ',')) { throw 'Citation coverage mismatch' }
foreach ($part in $parts) {
    if (([regex]::Matches($part,'(?m)^```').Count % 2) -ne 0) { throw 'Unbalanced fenced code block' }
    if ($part -match '\[CẦN BỔ SUNG\]|\bTODO\b|\bTBD\b|lorem ipsum|�') { throw 'Draft placeholder or encoding error found' }
    foreach ($caption in @([regex]::Matches($part,'(?m)^\*\*((?:Bảng|Hình) [^\r\n]+)\*\*$'))) {
        $following = $part.Substring($caption.Index+$caption.Length)
        $source = [regex]::Match($following,'(?m)^\*Nguồn:')
        $nextCaption = [regex]::Match($following,'(?m)^\*\*(?:Bảng|Hình) ')
        if (-not $source.Success -or ($nextCaption.Success -and $source.Index -gt $nextCaption.Index)) { throw "Missing caption source: $($caption.Value)" }
    }
}
$culture = [System.Globalization.CultureInfo]::GetCultureInfo('vi-VN')
$projectRoot = (Resolve-Path (Join-Path $AuditDirectory '../../../..')).Path
# tong-hop-so-lieu.json is committed and holds a copy of eval/results/baseline_test.json (git-ignored).
$payload = (Get-Content -LiteralPath (Join-Path $AuditDirectory 'tong-hop-so-lieu.json') -Raw -Encoding UTF8 | ConvertFrom-Json).results.baseline_test
$resultPart = $parts[4]
foreach ($metric in @('accuracy','spam_precision','spam_recall','spam_f1','roc_auc')) {
    $values = @('mlp','lstm','transformer') | ForEach-Object { (100*$payload.models.$_.$metric).ToString('F2',$culture)+'%' }
    $expected = '| ' + ($values -join ' | ') + ' |'
    if (-not $resultPart.Contains($expected)) { throw "Report metric row differs: $metric" }
}
foreach ($metric in @('latency_ms_batch_1','latency_ms_batch_32','train_seconds','parameters')) {
    $format = if ($metric -eq 'parameters') { 'N0' } elseif ($metric -eq 'train_seconds') { 'F3' } else { 'F4' }
    $values = @('mlp','lstm','transformer') | ForEach-Object { $payload.models.$_.$metric.ToString($format,$culture) }
    if (-not $resultPart.Contains('| '+($values -join ' | ')+' |')) { throw "Report cost row differs: $metric" }
}
foreach ($kind in @('mlp','lstm','transformer')) {
    $matrix = $payload.models.$kind.confusion_matrix
    $hamRow = "| HAM | $($matrix[0][0]) (TN) | $($matrix[0][1]) (FP) | 724 |"
    $spamRow = "| SPAM | $($matrix[1][0]) (FN) | $($matrix[1][1]) (TP) | 112 |"
    if (-not $resultPart.Contains($hamRow) -or -not $resultPart.Contains($spamRow)) { throw "Report matrix differs: $kind" }
}
$captionIds = @($captions | ForEach-Object { ([regex]::Match($_,'^(Bảng|Hình) ([^ ]+)')).Value })
if (@($captionIds | Sort-Object -Unique).Count -ne $captionIds.Count) { throw 'Duplicate caption identifiers' }
$tableCount = @($captions | Where-Object { $_ -like 'Bảng *' }).Count
$figureCount = @($captions | Where-Object { $_ -like 'Hình *' }).Count
$summary = @("CHAPTERS_OK: 9 parts present in order; combined body matches chapter files", "CITATIONS_OK: $($used.Count) references defined and cited; URL availability not verified", "CAPTIONS_OK: $tableCount tables and $figureCount figures with unique identifiers and source notes", 'MARKDOWN_OK: balanced code fences; no draft placeholders or replacement characters', 'STORED_METRIC_TABLES_OK: 5 metric rows, 4 cost rows and all 3 matrices match JSON', "SIZE: $($readback.Length) characters in combined report")
$summary | Set-Content -LiteralPath (Join-Path $AuditDirectory 'ket-qua-ra-soat.txt') -Encoding UTF8
$summary | Write-Output
