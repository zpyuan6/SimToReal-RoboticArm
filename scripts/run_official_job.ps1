param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [Parameter(Mandatory = $true)]
    [string]$JobCommand,
    [Parameter(Mandatory = $true)]
    [string]$LogPath,
    [switch]$Offline
)

$ErrorActionPreference = "Stop"

$logDir = Split-Path -Parent $LogPath
if ($logDir) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
}

Set-Location $RepoRoot
$env:UV_CACHE_DIR = Join-Path $RepoRoot ".uv-cache"
$env:HF_HOME = Join-Path $RepoRoot ".hf-home"
$env:TORCH_HOME = Join-Path $RepoRoot ".torch-home"
$env:PYTHONIOENCODING = "utf-8"
if ($Offline) {
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:HF_HUB_OFFLINE = "1"
}

$timestamp = Get-Date -Format o
"[$timestamp] START $JobCommand" | Out-File -FilePath $LogPath -Encoding utf8 -Append

try {
    Invoke-Expression $JobCommand *>> $LogPath
    $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }
}
catch {
    $_ | Out-File -FilePath $LogPath -Encoding utf8 -Append
    $exitCode = 1
}

$timestamp = Get-Date -Format o
"[$timestamp] EXIT $exitCode" | Out-File -FilePath $LogPath -Encoding utf8 -Append
exit $exitCode
