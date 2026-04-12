param(
    [string]$HostAddress = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$MarketInterval = 15,
    [int]$FuturesInterval = 60,
    [int]$CycleInterval = 3600
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ChartApp = Join-Path $RepoRoot "dashboards\chart_app"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

$apiArgs = @(
    (Join-Path $RepoRoot "api_server.py"),
    "--with-live-update",
    "--host", $HostAddress,
    "--port", $ApiPort,
    "--live-update-market-interval", $MarketInterval,
    "--live-update-futures-interval", $FuturesInterval,
    "--live-update-cycle-interval", $CycleInterval
)

$apiProcess = Start-Process -FilePath $PythonExe -ArgumentList $apiArgs -WorkingDirectory $RepoRoot -PassThru
try {
    npm.cmd run --prefix $ChartApp dev
}
finally {
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id
    }
}
