param([string]$TeXBin = '')
$ErrorActionPreference = 'Stop'
$compiler = if ($TeXBin) {
    Join-Path $TeXBin 'pdflatex.exe'
} elseif (Get-Command pdflatex.exe -ErrorAction SilentlyContinue) {
    (Get-Command pdflatex.exe).Source
} elseif (Test-Path -LiteralPath 'D:\texlive\2026\bin\windows\pdflatex.exe') {
    'D:\texlive\2026\bin\windows\pdflatex.exe'
} else {
    throw 'pdflatex.exe not found; pass -TeXBin or add TeX Live to PATH'
}
if (!(Test-Path -LiteralPath $compiler)) { throw "pdflatex not found: $compiler" }
$previousInputs = $env:TEXINPUTS
$env:TEXINPUTS = '../tex//;'
Push-Location (Join-Path $PSScriptRoot 'paper')
try {
    for ($pass = 1; $pass -le 3; $pass++) {
        & $compiler -interaction=nonstopmode -halt-on-error main.tex | Out-File -Encoding utf8 "build-pass-$pass.txt"
        if ($LASTEXITCODE -ne 0) { throw "LaTeX pass $pass failed; inspect paper/build-pass-$pass.txt" }
    }
    $log = Get-Content -Raw -LiteralPath 'main.log'
    if ($log -match 'undefined references|undefined citations|Overfull \\hbox|Overfull \\vbox') {
        throw 'Unresolved references or overflowing content; inspect paper/main.log'
    }
    Write-Output "Built $PSScriptRoot\paper\main.pdf"
} finally {
    Pop-Location
    $env:TEXINPUTS = $previousInputs
}
