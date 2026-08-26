[CmdletBinding()]
param(
    # [string]$Server = "root@77.91.79.118",
    [string]$Server = "test@100.65.83.101",
    [string]$RemoteDirectory = '${HOME}/pipestone'
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = $PSScriptRoot
$archiveName = "pipestone-deploy-$([Guid]::NewGuid().ToString('N')).tar.gz"
$localArchive = Join-Path ([System.IO.Path]::GetTempPath()) $archiveName
$remoteArchive = "/tmp/$archiveName"

function Resolve-NativeCommand {
    param(
        [string]$Name,
        [string[]]$FallbackPaths = @()
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    foreach ($fallbackPath in $FallbackPaths) {
        if ($fallbackPath -and (Test-Path -LiteralPath $fallbackPath -PathType Leaf)) {
            return $fallbackPath
        }
    }
    throw "Required command not found: $Name"
}

$windowsDirectory = [Environment]::GetEnvironmentVariable("WINDIR")
$sshCommand = Resolve-NativeCommand "ssh.exe" @(
    (Join-Path $windowsDirectory "System32\OpenSSH\ssh.exe"),
    "C:\Program Files\Git\usr\bin\ssh.exe"
)
$scpCommand = Resolve-NativeCommand "scp.exe" @(
    (Join-Path $windowsDirectory "System32\OpenSSH\scp.exe"),
    "C:\Program Files\Git\usr\bin\scp.exe"
)
$tarCommand = Resolve-NativeCommand "tar.exe" @(
    (Join-Path $windowsDirectory "System32\tar.exe")
)

$excludeArguments = @(
    "--exclude=.git",
    "--exclude=.vscode",
    "--exclude=__pycache__",
    "--exclude=.pytest_cache",
    "--exclude=.mypy_cache",
    "--exclude=output",
    "--exclude=run",
    "--exclude=*.pyc"
)

# In this project .env is a virtual environment directory, not a secrets file.
if (Test-Path (Join-Path $projectRoot ".env") -PathType Container) {
    $excludeArguments += "--exclude=.env"
}

function Assert-LastCommandSucceeded {
    param([string]$Operation)

    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

try {
    Write-Host "Creating project archive..."
    Push-Location $projectRoot
    try {
        & $tarCommand -czf $localArchive @excludeArguments .
        Assert-LastCommandSucceeded "Archive creation"
    }
    finally {
        Pop-Location
    }

    Write-Host "Creating $RemoteDirectory on $Server..."
    & $sshCommand $Server "mkdir -p -- `"$RemoteDirectory`""
    Assert-LastCommandSucceeded "Remote directory creation"

    Write-Host "Uploading files to $Server..."
    & $scpCommand $localArchive "${Server}:$remoteArchive"
    Assert-LastCommandSucceeded "Archive upload"

    Write-Host "Extracting files into $RemoteDirectory..."
    $remoteCommand = 'tar -xzf ''{0}'' -C "{1}"; deploy_status=$?; ' +
        'if [ "$deploy_status" -eq 0 ]; then chmod 600 "{1}/server_config.local.json" 2>/dev/null || true; fi; ' +
        'rm -f ''{0}''; exit "$deploy_status"'
    $remoteCommand = $remoteCommand -f $remoteArchive, $RemoteDirectory
    & $sshCommand $Server $remoteCommand
    Assert-LastCommandSucceeded "Archive extraction"

    Write-Host "Removing old project containers..."
    $restartCommand = 'cd "{0}" && docker compose down --remove-orphans && docker compose up --build -d && docker compose ps'
    $restartCommand = $restartCommand -f $RemoteDirectory
    & $sshCommand $Server $restartCommand
    Assert-LastCommandSucceeded "Remote Docker Compose rebuild"

    Write-Host "Done: project uploaded and started at ${Server}:$RemoteDirectory/"
}
finally {
    if (Test-Path -LiteralPath $localArchive) {
        Remove-Item -LiteralPath $localArchive -Force
    }
}
