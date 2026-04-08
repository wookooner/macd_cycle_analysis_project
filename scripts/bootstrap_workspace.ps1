param(
    [string]$DataRoot = "C:\Users\qw370\macd-cycle-data",
    [switch]$PersistUserEnv,
    [switch]$Validate
)

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:MACD_DATA_ROOT = $DataRoot
$env:PYTHONPATH = $repoRoot

$requiredDirs = @(
    $DataRoot,
    (Join-Path $DataRoot "raw"),
    (Join-Path $DataRoot "interim"),
    (Join-Path $DataRoot "processed"),
    (Join-Path $DataRoot "dashboard"),
    (Join-Path $DataRoot "outputs"),
    (Join-Path $DataRoot "reports"),
    (Join-Path $DataRoot "logs")
)

foreach ($dir in $requiredDirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

if ($PersistUserEnv) {
    [Environment]::SetEnvironmentVariable("MACD_DATA_ROOT", $DataRoot, "User")
}

Write-Host "Workspace bootstrap complete."
Write-Host "repoRoot         : $repoRoot"
Write-Host "MACD_DATA_ROOT   : $DataRoot"
Write-Host "persistedForUser : $PersistUserEnv"

if ($Validate) {
    python (Join-Path $repoRoot "scripts\validate_paths.py")
}
