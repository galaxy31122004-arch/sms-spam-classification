param(
    [string]$Python = (Join-Path $PSScriptRoot '..\..\..\..\eval\.venv\Scripts\python.exe')
)
# Full build of the Word report: render diagrams, convert Markdown, let Word build the fields,
# fix the styles Word generates, then repaginate and check the layout.
$ErrorActionPreference = 'Stop'
$python = (Resolve-Path $Python).Path
$here = $PSScriptRoot

& $python (Join-Path $here 'render-so-do.py')
& $python (Join-Path $here 'tao-ban-word.py')
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here 'cap-nhat-truong.ps1')
& $python (Join-Path $here 'ra-soat-ban-word.py') --fix-styles --fix-metadata
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here 'cap-nhat-truong.ps1')
& $python (Join-Path $here 'ra-soat-ban-word.py')
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $here 'kiem-tra-dan-trang.ps1')
