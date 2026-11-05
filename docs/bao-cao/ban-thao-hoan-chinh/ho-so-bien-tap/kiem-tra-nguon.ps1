param([string]$AuditDirectory = $PSScriptRoot)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $AuditDirectory '../../../..')).Path
$auditRoot = $AuditDirectory
Set-Location -LiteralPath $projectRoot
$sourcePaths = @('AGENTS.md','CLAUDE.md','README.md','PLAN.md','app.py','start_website.bat','test_model.py')
$sourcePaths += @(Get-ChildItem data,models,outputs,eval,tests -File -Recurse | Where-Object { $_.FullName -notmatch '[\\/](\.venv|__pycache__)[\\/]' } | ForEach-Object { $_.FullName.Substring($projectRoot.Length + 1).Replace('\','/') })
$sourcePaths += @('docs/bao-cao/01-cau-truc-tong-the.md','docs/bao-cao/02-yeu-cau-thong-tin.md','docs/bao-cao/03-mo-dau.md')
$hashes = @($sourcePaths | Sort-Object -Unique | ForEach-Object { [ordered]@{path=$_; sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash} })
$hashPath = Join-Path $auditRoot 'source-hashes.json'
if (-not (Test-Path -LiteralPath $hashPath)) { $hashes | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $hashPath -Encoding UTF8 }
$original = Get-Content -LiteralPath $hashPath -Raw -Encoding UTF8 | ConvertFrom-Json
$changed = @($original | Where-Object { -not (Test-Path -LiteralPath $_.path) -or (Get-FileHash -LiteralPath $_.path -Algorithm SHA256).Hash -ne $_.sha256 })
if ($changed.Count) { throw 'A protected source file changed.' }
$splits = @{}
$counts = [ordered]@{}
foreach ($name in @('train','val','test')) {
    $rows = @(Import-Csv -LiteralPath "data/splits/$name.csv" -Encoding UTF8)
    $splits[$name] = $rows
    $counts[$name] = [ordered]@{rows=$rows.Count; ham=@($rows | Where-Object label -CEQ 'ham').Count; spam=@($rows | Where-Object label -CEQ 'spam').Count; columns=@($rows[0].PSObject.Properties.Name)}
}
$raw = @(Import-Csv -LiteralPath 'data/raw/spam.csv' -Encoding Default)
$counts['raw'] = [ordered]@{rows=$raw.Count; ham=@($raw | Where-Object v1 -CEQ 'ham').Count; spam=@($raw | Where-Object v1 -CEQ 'spam').Count; columns=@($raw[0].PSObject.Properties.Name)}
$overlap = @()
foreach ($field in @('raw_text','text')) {
    foreach ($pair in @(@('train','val'),@('train','test'),@('val','test'))) {
        $seen = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
        foreach ($row in $splits[$pair[0]]) { [void]$seen.Add($row.$field) }
        $repeated = @($splits[$pair[1]] | Where-Object { $seen.Contains($_.$field) })
        $groups = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::Ordinal)
        foreach ($row in $repeated) { [void]$groups.Add($row.$field) }
        $overlap += [ordered]@{field=$field; left=$pair[0]; right=$pair[1]; shared_groups=$groups.Count; rows_in_right=$repeated.Count; spam_rows_in_right=@($repeated | Where-Object label -CEQ 'spam').Count}
    }
}
$payload = Get-Content outputs/test_results.json -Raw -Encoding UTF8 | ConvertFrom-Json
$metricChecks = @()
foreach ($kind in @('mlp','lstm','transformer')) {
    $m = $payload.models.$kind
    $tn = [double]$m.confusion_matrix[0][0]; $fp = [double]$m.confusion_matrix[0][1]
    $fn = [double]$m.confusion_matrix[1][0]; $tp = [double]$m.confusion_matrix[1][1]
    $computed = [ordered]@{accuracy=($tn+$tp)/($tn+$fp+$fn+$tp); spam_precision=$tp/($tp+$fp); spam_recall=$tp/($tp+$fn); spam_f1=2*$tp/(2*$tp+$fp+$fn)}
    foreach ($key in $computed.Keys) { if ([Math]::Abs($computed[$key]-$m.$key) -gt 1e-12) { throw "Metric mismatch: $kind $key" } }
    if ($tn+$fp -ne 724 -or $fn+$tp -ne 112) { throw 'Matrix label totals mismatch' }
    $config = Get-Content "outputs/$kind/config.json" -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($property in $m.config.PSObject.Properties) { if ($config.($property.Name) -ne $property.Value) { throw "Config mismatch: $kind $($property.Name)" } }
    if ($m.parameters -ne $config.parameters -or $m.train_seconds -ne $config.train_seconds) { throw 'Cost/config mismatch' }
    $metricChecks += [ordered]@{model=$kind; recomputed=$computed; consistent=$true}
}
if ($counts.raw.rows -ne 5572 -or $counts.train.rows -ne 3900 -or $counts.val.rows -ne 836 -or $counts.test.rows -ne 836) { throw 'Unexpected dataset counts' }
$audit = [ordered]@{method='PowerShell Import-Csv; exact ordinal string matching; arithmetic from stored confusion matrices; no model inference'; counts=$counts; overlap=$overlap; metric_checks=$metricChecks; protected_sources=$original.Count; source_integrity='UNCHANGED'; python_tests='NOT RUN: Python launcher has no installed interpreter; project venv base interpreter is missing'; roc_auc='Read from stored JSON only; not recomputed'; checkpoint_binaries='Present and hashed; not loaded'}
$audit | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $auditRoot 'kiem-tra-du-lieu.json') -Encoding UTF8
Write-Output "CSV_COUNTS_OK: raw=$($counts.raw.rows), train=$($counts.train.rows), val=$($counts.val.rows), test=$($counts.test.rows)"
$trainTestOverlap = @($overlap | Where-Object { $_.field -eq 'text' -and $_.left -eq 'train' -and $_.right -eq 'test' })[0]
Write-Output "TEXT_TRAIN_TEST_OVERLAP: $($trainTestOverlap.rows_in_right) test rows"
Write-Output 'METRICS_OK: Accuracy, Precision, Recall, F1 for all 3 models; config fields and saved costs agree'
Write-Output "SOURCE_INTEGRITY_OK: $($original.Count) files unchanged"
