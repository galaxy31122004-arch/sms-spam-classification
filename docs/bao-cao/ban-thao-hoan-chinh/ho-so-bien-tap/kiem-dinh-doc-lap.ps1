param([string]$AuditDirectory = $PSScriptRoot, [ValidateSet('before','after')][string]$Phase = 'after')
$ErrorActionPreference = 'Stop'
$reportRoot = (Resolve-Path (Join-Path $AuditDirectory '..')).Path
$projectRoot = (Resolve-Path (Join-Path $AuditDirectory '../../../..')).Path
$files = @('01-mo-dau.md','02-chuong-1-co-so-ly-thuyet.md','03-chuong-2-du-lieu-va-tien-xu-ly.md','04-chuong-3-mo-hinh-va-huan-luyen.md','05-chuong-4-ket-qua-va-so-sanh.md','06-chuong-5-website-demo.md','07-ket-luan.md','08-tai-lieu-tham-khao.md','09-phu-luc.md')
$anchors = @('mo-dau','chuong-1','chuong-2','chuong-3','chuong-4','chuong-5','ket-luan','tai-lieu-tham-khao','phu-luc')
$parts = @($files | ForEach-Object { [IO.File]::ReadAllText((Join-Path $reportRoot $_)).Replace("`r`n","`n").TrimEnd() })
$combined = [IO.File]::ReadAllText((Join-Path $reportRoot 'BAO-CAO-TONG-HOP.md')).Replace("`r`n","`n")
$matches = [regex]::Matches($combined,'(?ms)^<a id="(?<anchor>[^"]+)"></a>\n\n(?<body>.*?)(?=\n\n---\n\n<a id=|\z)')
if ($matches.Count -ne 9) { throw 'Combined section count is not nine' }
$sectionChecks = @()
for ($i=0; $i -lt 9; $i++) {
    $actual = $matches[$i].Groups['body'].Value.TrimEnd()
    if ($matches[$i].Groups['anchor'].Value -cne $anchors[$i] -or $actual -cne $parts[$i]) { throw "Existing combined report differs at $($files[$i])" }
    $sectionChecks += [ordered]@{file=$files[$i]; characters=$parts[$i].Length; exact_body_match=$true; sha256=(Get-FileHash -LiteralPath (Join-Path $reportRoot $files[$i]) -Algorithm SHA256).Hash}
}
function Read-ReportTable([string]$Text,[string]$Caption) {
    $start = $Text.IndexOf("**$Caption.",[StringComparison]::Ordinal)
    if ($start -lt 0) { throw "Missing table $Caption" }
    $tail = $Text.Substring($start)
    $end = $tail.IndexOf('*Nguồn:',[StringComparison]::Ordinal)
    if ($end -lt 0) { throw "Missing source $Caption" }
    $rows = [regex]::Matches($tail.Substring(0,$end),'(?m)^\|([^\r\n]+)\|\s*$')
    $result = @{}
    foreach ($row in $rows) {
        $cells = @($row.Groups[1].Value.Split('|') | ForEach-Object { $_.Trim() })
        if ($cells[0] -notmatch '^[-:]') {
            $codeKey = [regex]::Match($cells[0],'^`([^`]+)`')
            $key = if ($codeKey.Success) { $codeKey.Groups[1].Value } else { $cells[0] }
            $result[$key] = $cells[1..($cells.Count-1)]
        }
    }
    return $result
}
$culture = [Globalization.CultureInfo]::GetCultureInfo('vi-VN')
$payload = Get-Content -LiteralPath (Join-Path $projectRoot 'outputs/test_results.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$metricRows = Read-ReportTable $parts[4] 'Bảng 4.2'
$costRows = Read-ReportTable $parts[4] 'Bảng 4.6'
$configRows = Read-ReportTable $parts[8] 'Bảng PL.B.1'
$metricNames = [ordered]@{accuracy='Accuracy';spam_precision='Precision SPAM';spam_recall='Recall SPAM';spam_f1='F1 SPAM';roc_auc='ROC-AUC'}
$metricChecks = @()
$index = 0
foreach ($kind in @('mlp','lstm','transformer')) {
    $m = $payload.models.$kind
    foreach ($key in $metricNames.Keys) { if ($metricRows[$metricNames[$key]][$index] -cne ((100*$m.$key).ToString('F2',$culture)+'%')) { throw "Metric table mismatch $kind $key" } }
    foreach ($cost in @(@('parameters','Số tham số mạng','N0'),@('train_seconds','Thời gian huấn luyện (giây)','F3'),@('latency_ms_batch_1','Latency batch 1 (ms/tin)','F4'),@('latency_ms_batch_32','Latency batch 32 (ms/tin)','F4'))) {
        if ($costRows[$cost[1]][$index] -cne $m.($cost[0]).ToString($cost[2],$culture)) { throw "Cost table mismatch $kind $($cost[0])" }
    }
    $tn=[double]$m.confusion_matrix[0][0]; $fp=[double]$m.confusion_matrix[0][1]; $fn=[double]$m.confusion_matrix[1][0]; $tp=[double]$m.confusion_matrix[1][1]
    $computed = [ordered]@{accuracy=($tn+$tp)/($tn+$fp+$fn+$tp);spam_precision=$tp/($tp+$fp);spam_recall=$tp/($tp+$fn);spam_f1=2*$tp/(2*$tp+$fp+$fn)}
    foreach ($key in $computed.Keys) { if ([Math]::Abs($computed[$key]-$m.$key) -gt 1e-12) { throw "Stored arithmetic mismatch $kind $key" } }
    $matrixRows = Read-ReportTable $parts[4] ('Bảng 4.'+(3+$index))
    if (($matrixRows['HAM'] -join '|') -cne "$tn (TN)|$fp (FP)|724" -or ($matrixRows['SPAM'] -join '|') -cne "$fn (FN)|$tp (TP)|112") { throw "Matrix mismatch $kind" }
    $config = Get-Content -LiteralPath (Join-Path $projectRoot "outputs/$kind/config.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($prop in $config.PSObject.Properties) {
        $cell = $configRows[$prop.Name][$index].Trim('`')
        if ($null -eq $prop.Value) { if ($cell -cne 'n/a') { throw "Null config mismatch $kind $($prop.Name)" } }
        elseif ($prop.Value -is [string]) { if ($cell -cne $prop.Value) { throw "String config mismatch $kind $($prop.Name)" } }
        else { $number=[double]::Parse($cell,[Globalization.NumberStyles]::Number,$culture); if ([Math]::Abs($number-[double]$prop.Value) -gt 1e-12) { throw "Numeric config mismatch $kind $($prop.Name)" } }
    }
    foreach ($prop in $m.config.PSObject.Properties) { if ($config.($prop.Name) -ne $prop.Value) { throw "Stored config mismatch $kind $($prop.Name)" } }
    $metricChecks += [ordered]@{model=$kind; arithmetic=$computed; report_metrics_and_costs_match=$true; full_config_fields_checked=$config.PSObject.Properties.Name.Count}
    $index++
}
$splits = @{}; $counts = @()
foreach ($name in @('train','val','test')) {
    $rows = @(Import-Csv -LiteralPath (Join-Path $projectRoot "data/splits/$name.csv") -Encoding UTF8)
    $splits[$name]=$rows
    $counts += [ordered]@{name=$name;rows=$rows.Count;ham=@($rows | Where-Object label -CEQ 'ham').Count;spam=@($rows | Where-Object label -CEQ 'spam').Count}
}
$raw = @(Import-Csv -LiteralPath (Join-Path $projectRoot 'data/raw/spam.csv') -Encoding Default)
$counts += [ordered]@{name='raw';rows=$raw.Count;ham=@($raw | Where-Object v1 -CEQ 'ham').Count;spam=@($raw | Where-Object v1 -CEQ 'spam').Count}
$dataRows=Read-ReportTable $parts[2] 'Bảng 2.1'
$dataNames=@{raw='Toàn bộ dữ liệu';train='Train';val='Validation';test='Test'}
foreach ($count in $counts) {
    $percentage=(100.0*$count.spam/$count.rows).ToString('F2',$culture)+'%'
    $expected=@($count.rows.ToString('N0',$culture),$count.ham.ToString('N0',$culture),$count.spam.ToString('N0',$culture),$percentage)
    if (($dataRows[$dataNames[$count.name]] -join '|') -cne ($expected -join '|')) { throw "Dataset table mismatch $($count.name)" }
}
$overlaps=@()
foreach ($field in @('raw_text','text')) {
    foreach ($pair in @(@('train','val'),@('train','test'),@('val','test'))) {
        $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($r in $splits[$pair[0]]) { [void]$seen.Add($r.$field) }
        $repeated=@($splits[$pair[1]] | Where-Object { $seen.Contains($_.$field) })
        $unique=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
        foreach ($r in $repeated) { [void]$unique.Add($r.$field) }
        $overlaps += [ordered]@{field=$field;left=$pair[0];right=$pair[1];distinct_texts=$unique.Count;rows=$repeated.Count;spam_rows=@($repeated | Where-Object label -CEQ 'spam').Count}
    }
}
$splitLabels=@{train='Train';val='Validation';test='Test'}
foreach ($overlap in $overlaps) {
    $expected='| `'+$overlap.field+'` | '+$splitLabels[$overlap.left]+' → '+$splitLabels[$overlap.right]+' | '+$overlap.distinct_texts+' | '+$overlap.rows+' | '+$overlap.spam_rows+' |'
    if (-not $parts[2].Contains($expected)) { throw 'Duplicate table mismatch' }
}
$body=($parts[0..6]+$parts[8]) -join "`n"
$defined=@([regex]::Matches($parts[7],'(?m)^\[(\d+)\] ') | ForEach-Object { [int]$_.Groups[1].Value })
$used=@([regex]::Matches($body,'\[(\d+)\]') | ForEach-Object { [int]$_.Groups[1].Value } | Sort-Object -Unique)
if (($defined -join ',') -ne '1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16' -or ($used -join ',') -ne ($defined -join ',')) { throw 'Citation IDs do not match' }
$math=@()
foreach ($i in 0..8) {
    $blocks=[regex]::Matches($parts[$i],'(?ms)^\$\$\n(.*?)\n\$\$')
    if ([regex]::Matches($parts[$i],'(?m)^\$\$$').Count -ne 2*$blocks.Count) { throw 'Math delimiter mismatch' }
    foreach ($block in $blocks) {
        $formula=$block.Groups[1].Value
        $braceDepth=0
        for ($j=0;$j -lt $formula.Length;$j++) {
            if ($j -gt 0 -and $formula[$j-1] -eq '\') { continue }
            if ($formula[$j] -eq '{') { $braceDepth++ }
            if ($formula[$j] -eq '}') { $braceDepth--; if ($braceDepth -lt 0) { throw 'Unexpected math closing brace' } }
        }
        if ($braceDepth -ne 0) { throw 'Unbalanced math braces' }
        foreach ($environment in @('aligned','cases')) { if ([regex]::Matches($formula,'\\begin\{'+$environment+'\}').Count -ne [regex]::Matches($formula,'\\end\{'+$environment+'\}').Count) { throw 'Unbalanced math environment' } }
        $math += [ordered]@{file=$files[$i];tag=([regex]::Match($formula,'\\tag\{([^}]+)\}')).Groups[1].Value;formula=$formula}
    }
}
$protected=Get-Content -LiteralPath (Join-Path $AuditDirectory 'source-hashes.json') -Raw -Encoding UTF8 | ConvertFrom-Json
foreach ($entry in $protected) { if ((Get-FileHash -LiteralPath (Join-Path $projectRoot $entry.path) -Algorithm SHA256).Hash -cne $entry.sha256) { throw "Protected source changed: $($entry.path)" } }
$report = [ordered]@{phase=$Phase;combined_sections=$sectionChecks;metric_and_config_checks=$metricChecks;counts=$counts;overlaps=$overlaps;citation_ids=$defined;display_math=$math;protected_source_count=$protected.Count;protected_sources_unchanged=$true;limitations=@('No model inference, training, split generation, or application tests executed','Math checked as source and by manual reasoning; no visual math rendering','Reference identity and source support require separate manual verification; matching IDs is insufficient')}
$report | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $AuditDirectory "kiem-dinh-$Phase.json") -Encoding UTF8
Write-Output "INDEPENDENT_$($Phase.ToUpper())_CHECK: existing combined report has 9 exact ordered sections; no report rewrite by this checker"
Write-Output 'TABLE_CHECK: dataset/duplicate rows, named metric/cost rows, each matrix, all full config fields agree with stored sources'
Write-Output "SOURCE_CHECK: $($protected.Count) protected source files unchanged; display math blocks=$($math.Count); cited reference IDs=$($used.Count)"
