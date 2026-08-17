param(
    [int]$Port = 9222,
    [string]$Url = "https://spb.hh.ru/search/vacancy?resume=5cef9fa3ff080d093b0039ed1f456156794c54&hhtmFromLabel=tab_byResume&hhtmFrom=main"
)

$candidates = @(
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)
$chrome = $candidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $chrome) {
    throw "Google Chrome не найден. Укажите путь к chrome.exe вручную."
}

$profile = Join-Path $PSScriptRoot ".chrome-profile"
New-Item -ItemType Directory -Force -Path $profile | Out-Null

Start-Process -FilePath $chrome -ArgumentList @(
    "--remote-debugging-port=$Port",
    "--user-data-dir=$profile",
    $Url
)

Write-Host "Chrome запущен на DevTools-порту $Port."
Write-Host "После входа в hh.ru запустите парсер с --cdp-url http://127.0.0.1:$Port"
