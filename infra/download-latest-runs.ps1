[CmdletBinding()]
param(
    [string]$Server = "root@77.91.79.118",
    [string]$RemoteRunDirectory = '${HOME}/pipestone/run',
    [string]$LocalRunDirectory = (Join-Path $PSScriptRoot "run"),
    [ValidateRange(1, 100)]
    [int]$Count = 5
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$archiveName = "pipestone-runs-$([Guid]::NewGuid().ToString('N')).tar.gz"
$localArchive = Join-Path ([System.IO.Path]::GetTempPath()) $archiveName
$remoteArchive = "/tmp/$archiveName"
$remoteArchiveCreated = $false

foreach ($commandName in @("ssh.exe", "scp.exe", "tar.exe")) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Required command not found: $commandName"
    }
}

if ($RemoteRunDirectory.IndexOfAny(@([char]'"', [char]"`r", [char]"`n")) -ge 0) {
    throw "RemoteRunDirectory contains unsupported characters."
}

function Assert-LastCommandSucceeded {
    param([string]$Operation)

    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

try {
    New-Item -ItemType Directory -Path $LocalRunDirectory -Force | Out-Null

    Write-Host "Selecting the latest $Count run directories on $Server..."
    $remoteCommand = @'
set -eu
run_dir="{0}"
archive="{1}"
if [ ! -d "$run_dir" ]; then
    echo "Remote run directory does not exist: $run_dir" >&2
    exit 3
fi
cd "$run_dir"
latest="$(find . -mindepth 1 -maxdepth 1 -type d -printf '%T@ %P\n' | sort -nr | head -n {2} | cut -d' ' -f2-)"
if [ -z "$latest" ]; then
    echo "No run directories found in $run_dir" >&2
    exit 4
fi
printf '%s\n' "$latest" | tar --verbatim-files-from -czf "$archive" -T -
printf '%s\n' "$latest"
'@ -f $RemoteRunDirectory, $remoteArchive, $Count

    $downloadedDirectories = @(& ssh.exe $Server $remoteCommand)
    Assert-LastCommandSucceeded "Remote archive creation"
    $remoteArchiveCreated = $true

    Write-Host "Downloading archive from $Server..."
    & scp.exe "${Server}:$remoteArchive" $localArchive
    Assert-LastCommandSucceeded "Archive download"

    Write-Host "Extracting runs into $LocalRunDirectory..."
    & tar.exe -xzf $localArchive -C $LocalRunDirectory
    Assert-LastCommandSucceeded "Archive extraction"

    Write-Host "Downloaded run directories:"
    $downloadedDirectories | ForEach-Object { Write-Host "  $_" }
}
finally {
    if (Test-Path -LiteralPath $localArchive) {
        Remove-Item -LiteralPath $localArchive -Force
    }
    if ($remoteArchiveCreated) {
        & ssh.exe $Server "rm -f -- '$remoteArchive'" 2>$null | Out-Null
    }
}
