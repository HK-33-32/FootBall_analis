param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$WeightsBundle,
    [string]$Sha256 = "",
    [switch]$Replace,
    [switch]$NoStart
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$arguments = @("scripts/weights_bundle.py", "install", $WeightsBundle)
if ($Sha256) { $arguments += @("--archive-sha256", $Sha256) }
if ($Replace) { $arguments += "--replace" }
python @arguments
if ($LASTEXITCODE -ne 0) { throw "Weight bundle installation failed" }
if (-not $NoStart) { & "$PSScriptRoot/quickstart-gpu.ps1" }
