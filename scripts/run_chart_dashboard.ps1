$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ChartApp = Join-Path $RepoRoot "dashboards\chart_app"

npm.cmd run --prefix $ChartApp dev
