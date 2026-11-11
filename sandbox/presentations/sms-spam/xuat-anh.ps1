param([string]$Path, [string]$Out)
$ErrorActionPreference = 'Stop'
$app = New-Object -ComObject PowerPoint.Application
try {
    $p = $app.Presentations.Open((Resolve-Path $Path).Path, $true, $false, $false)
    New-Item -ItemType Directory -Force $Out | Out-Null
    $i = 1
    foreach ($s in $p.Slides) { $s.Export((Join-Path (Resolve-Path $Out).Path ("slide{0:D2}.png" -f $i)), 'PNG', 1280, 720); $i++ }
    $p.Close()
    "Exported $($i-1)"
} finally { $app.Quit() }
