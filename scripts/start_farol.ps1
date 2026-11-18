#requires -Version 5.1
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$Projects = Join-Path $Repo "data\projects"
$Port = 8765
$FarolHost = "127.0.0.1"
$BaseUrl = "http://${FarolHost}:${Port}"

function Show-LauncherError($message) {
    Write-Host ""
    Write-Host "Farol: $message" -ForegroundColor Red
    Write-Host "Log: $script:ServerLog" -ForegroundColor DarkGray
    Read-Host "Pressione Enter para fechar"
    exit 1
}

function Get-FarolStatus {
    try {
        $response = Invoke-WebRequest -Uri "$BaseUrl/api/projects" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            try {
                $payload = $response.Content | ConvertFrom-Json
                if ($null -ne $payload.projects) { return "farol" }
            } catch {
                return "other"
            }
        }
        return "other"
    } catch {
        if ($_.Exception.Response) { return "other" }
        return "down"
    }
}

$script:ServerLog = Join-Path $env:TEMP "farol-server.log"
$serverErrorLog = Join-Path $env:TEMP "farol-server-error.log"

$status = Get-FarolStatus
if ($status -eq "farol") {
    Start-Process $BaseUrl
    exit 0
}

$portBusy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($status -eq "other" -or $portBusy) {
    Show-LauncherError "a porta $Port já está ocupada por outro serviço. Feche o serviço ou altere a porta no lançador."
}

$python = $null
$pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
if ($pythonCommand) { $python = $pythonCommand.Source }
if (-not $python) {
    Show-LauncherError "o interpretador python.exe não foi encontrado no PATH."
}
if (-not (Test-Path $Repo)) {
    Show-LauncherError "o diretório do repositório não existe: $Repo"
}
New-Item -ItemType Directory -Force -Path $Projects | Out-Null

$arguments = @(
    "-m", "sinapse", "web",
    "--root", "`"$Projects`"",
    "--host", $FarolHost,
    "--port", "$Port"
)
$process = Start-Process -FilePath $python -ArgumentList $arguments `
    -WorkingDirectory $Repo -WindowStyle Hidden `
    -RedirectStandardOutput $script:ServerLog `
    -RedirectStandardError $serverErrorLog -PassThru

for ($attempt = 0; $attempt -lt 60; $attempt++) {
    Start-Sleep -Milliseconds 250
    if ((Get-FarolStatus) -eq "farol") {
        Start-Process $BaseUrl
        exit 0
    }
    if ($process.HasExited) { break }
}

if (-not $process.HasExited) {
    Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
}
$details = @()
if (Test-Path $script:ServerLog) { $details += (Get-Content $script:ServerLog -Tail 20) }
if (Test-Path $serverErrorLog) { $details += (Get-Content $serverErrorLog -Tail 20) }
if ($details.Count -gt 0) { $details | ForEach-Object { Write-Host $_ -ForegroundColor DarkGray } }
Show-LauncherError "não foi possível iniciar o servidor em $BaseUrl."
