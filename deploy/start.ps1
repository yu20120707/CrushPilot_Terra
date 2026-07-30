param([switch]$Rebuild)

$ErrorActionPreference = "Stop"

$composeDir = $PSScriptRoot
$dockerCandidates = @(
    (Get-Command docker -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
    "E:\Docker\DockerDesktop\resources\bin\docker.exe",
    "C:\Program Files\Docker\Docker\resources\bin\docker.exe"
) | Where-Object { $_ -and (Test-Path $_) }

if (-not $dockerCandidates) {
    throw "Docker Desktop was not found. Install and start it, then retry."
}

$docker = @($dockerCandidates)[0]
$env:PATH = "$(Split-Path -Parent $docker);$env:PATH"
$envFile = Join-Path $composeDir ".env"
if (-not (Test-Path $envFile)) {
    throw "Missing deploy/.env. Copy .env.example and set the model configuration."
}

function Wait-DockerDesktop {
    $deadline = (Get-Date).AddMinutes(3)
    do {
        $status = (& $docker desktop status 2>$null | Out-String)
        if ($status -match "running") { return }
        Start-Sleep -Seconds 5
    } while ((Get-Date) -lt $deadline)
    throw "Docker Desktop did not become ready within three minutes."
}

$desktopStatus = (& $docker desktop status 2>$null | Out-String)
if ($desktopStatus -notmatch "running") {
    Write-Host "Starting Docker Desktop..."
    & $docker desktop start
    Wait-DockerDesktop
}

Push-Location $composeDir
try {
    if ($Rebuild) {
        & $docker compose --env-file .env up --build -d
    }
    else {
        & $docker compose --env-file .env up -d
    }
    if ($LASTEXITCODE -ne 0) { throw "Compose startup failed." }

    $deadline = (Get-Date).AddMinutes(10)
    do {
        $backendId = (& $docker compose --env-file .env ps -q backend).Trim()
        $health = if ($backendId) { (& $docker inspect -f '{{.State.Health.Status}}' $backendId).Trim() }
        if ($health -eq "healthy") { break }
        Start-Sleep -Seconds 5
    } while ((Get-Date) -lt $deadline)

    if ($health -ne "healthy") {
        & $docker compose --env-file .env ps
        throw "Backend did not become healthy within ten minutes. Run docker compose logs backend corpus-publisher."
    }

    & $docker compose --env-file .env ps
    Write-Host "Startup complete:"
    Write-Host "  Frontend: http://[::1]:8080/"
    Write-Host "  Backend:  http://[::1]:8080/ready"
}
finally {
    Pop-Location
}
