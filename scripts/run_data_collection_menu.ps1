$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$MenuScript = Join-Path $ScriptDir "data_collection_menu.py"

if (-not (Test-Path -LiteralPath $Python)) {
    $Python = "python"
}

& $Python $MenuScript @args
exit $LASTEXITCODE
