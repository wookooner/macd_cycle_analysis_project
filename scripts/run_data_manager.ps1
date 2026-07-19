param(
    [string]$HostAddress = "127.0.0.1",
    [int]$ApiPort = 8000,
    [int]$ManagerPort = 5174
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ChartApp = Join-Path $RepoRoot "dashboards\chart_app"
$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) { $PythonExe = "python" }

# Reuse the chart API when it is already running. Otherwise this launcher owns
# a lightweight local API process and stops it when the manager closes.
$existingApi = Get-NetTCPConnection -State Listen -LocalPort $ApiPort -ErrorAction SilentlyContinue
$apiProcess = $null
if (-not $existingApi) {
    $apiArgs = @((Join-Path $RepoRoot "api_server.py"), "--host", $HostAddress, "--port", $ApiPort)
    $apiProcess = Start-Process -FilePath $PythonExe -ArgumentList $apiArgs -WorkingDirectory $RepoRoot -PassThru -WindowStyle Hidden
}

try {
    npm.cmd run --prefix $ChartApp dev:manager -- --port $ManagerPort
}
finally {
    if ($apiProcess -and -not $apiProcess.HasExited) {
        Stop-Process -Id $apiProcess.Id
    }
}
