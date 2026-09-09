$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$weights = if ($env:FI_WEIGHTS_DIR) { $env:FI_WEIGHTS_DIR } else { ".\weights" }
$required = @(
    "CLIP_Jersey.pth",
    "SoccernetGSR_EfficientNet_Best.pth",
    "sports_model.pth.tar-60"
)
$optional = @("Qwen2.5-VL-7B-Instruct-Q8_0.gguf", "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf")

docker run --rm --gpus all --entrypoint nvidia-smi `
    nvidia/cuda:12.8.1-base-ubuntu24.04 | Out-Null
if (-not (Test-Path -LiteralPath $weights -PathType Container)) {
    throw "Weights directory does not exist: $weights"
}
$missing = foreach ($name in $required) {
    if (-not (Get-ChildItem -LiteralPath $weights -Recurse -File -Filter $name -ErrorAction SilentlyContinue)) {
        $name
    }
}
if ($missing) {
    throw "Missing required checkpoint files under $weights`: $($missing -join ', ')"
}
$missingOptional = foreach ($name in $optional) {
    if (-not (Get-ChildItem -LiteralPath $weights -Recurse -File -Filter $name -ErrorAction SilentlyContinue)) {
        $name
    }
}
Write-Host "GPU and required checkpoint filenames are available."
if ($missingOptional) {
    Write-Host "Optional local Qwen files absent; jersey reading will use CLIP: $($missingOptional -join ', ')"
}
Write-Host 'Start with: $env:FI_CORE_URL="http://perception:8000"; $env:FI_PERCEPTION_URL="http://perception:8000"; docker compose --profile full up -d --build api perception'
