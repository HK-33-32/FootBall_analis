$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is required: https://docs.docker.com/desktop/"
}
docker compose version | Out-Null
if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
}
foreach ($directory in @("weights/checkpoints", "data/runtime", "data/uploads", "runs")) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

Write-Host "Checking Docker GPU access..."
docker run --rm --gpus all --entrypoint nvidia-smi `
    nvidia/cuda:12.8.1-base-ubuntu24.04 | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker cannot access an NVIDIA GPU" }

Write-Host "Building the integrated perception service (first build is large)..."
docker compose --profile full build perception
if ($LASTEXITCODE -ne 0) { throw "Perception build failed" }

Write-Host "Downloading missing public core checkpoints..."
docker compose --profile full run --rm perception fetch-weights
if ($LASTEXITCODE -ne 0) { throw "Required checkpoint download/check failed" }
& "$PSScriptRoot/check_gpu_prerequisites.ps1"

$env:FI_CORE_URL = "http://perception:8000"
$env:FI_PERCEPTION_URL = "http://perception:8000"
docker compose --profile full up -d --build api perception
if ($LASTEXITCODE -ne 0) { throw "Compose startup failed" }

for ($attempt = 0; $attempt -lt 120; $attempt++) {
    try {
        $core = Invoke-RestMethod -Uri "http://127.0.0.1:8000/v1/health" -TimeoutSec 5
        $app = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/health" -TimeoutSec 5
        if ($core.status -eq "ok" -and $app.status -eq "ok") {
            Write-Host "Full stack is ready: http://localhost:8080"
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}
docker compose logs --tail 150 api perception
throw "Full stack did not become healthy within 240 seconds"
