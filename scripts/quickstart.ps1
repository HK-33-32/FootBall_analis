$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker Desktop is required: https://docs.docker.com/desktop/"
}
docker compose version | Out-Null

if (-not (Test-Path -LiteralPath ".env")) {
    Copy-Item -LiteralPath ".env.example" -Destination ".env"
    Write-Host "Created .env from .env.example"
}
foreach ($directory in @("data/runtime", "data/uploads", "runs", "weights")) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

docker compose up -d --build api
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8080/api/v1/health" -TimeoutSec 5
        if ($health.status -eq "ok") {
            Write-Host "Football Intelligence is ready: http://localhost:8080"
            exit 0
        }
    } catch {
        Start-Sleep -Seconds 2
    }
}
docker compose logs --tail 100 api
throw "API did not become healthy within 120 seconds"
