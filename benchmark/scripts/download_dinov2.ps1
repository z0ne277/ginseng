[CmdletBinding()]
param(
    [string]$OutputDirectory = "",
    [switch]$Force,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ModelId = "facebook/dinov2-base"
$Revision = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
$WeightLength = 346345912
$WeightSha256 = "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"

if (-not $OutputDirectory) {
    $OutputDirectory = Join-Path $RepoRoot "artifacts\models\dinov2_base"
} elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
    $OutputDirectory = Join-Path $RepoRoot $OutputDirectory
}
$OutputDirectory = [IO.Path]::GetFullPath($OutputDirectory)

function Test-DownloadedFile {
    param(
        [string]$Path,
        [long]$ExpectedLength = 0,
        [string]$ExpectedSha256 = ""
    )
    $item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    if (-not $item -or $item.PSIsContainer -or $item.Length -le 0) { return $false }
    if ($ExpectedLength -gt 0 -and $item.Length -ne $ExpectedLength) { return $false }
    if ($ExpectedSha256) {
        $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $ExpectedSha256.ToLowerInvariant()) { return $false }
    }
    return $true
}

function Invoke-PinnedDownload {
    param(
        [string]$Name,
        [long]$ExpectedLength = 0,
        [string]$ExpectedSha256 = ""
    )
    $target = Join-Path $OutputDirectory $Name
    $partial = "$target.part"
    $url = "https://huggingface.co/$ModelId/resolve/$Revision/$Name"

    if (Test-DownloadedFile $target $ExpectedLength $ExpectedSha256) {
        Write-Host "[dinov2:download] verified existing $Name"
        return
    }
    if (Test-Path -LiteralPath $target) {
        if (-not $Force) {
            throw "Invalid existing file: $target. Re-run with -Force to replace it."
        }
        Remove-Item -LiteralPath $target -Force
    }
    if ($Force -and (Test-Path -LiteralPath $partial)) {
        Remove-Item -LiteralPath $partial -Force
    }

    $curlArguments = @(
        "-L", "--fail", "--retry", "8", "--retry-delay", "5",
        "--connect-timeout", "30", "--speed-time", "60", "--speed-limit", "1024",
        "-C", "-", "--progress-bar", "-o", $partial, $url
    )
    Write-Host "[dinov2:download] curl.exe $($curlArguments -join ' ')"
    if ($DryRun) { return }

    & curl.exe @curlArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $Name with curl exit code $LASTEXITCODE. The .part file is retained for resume."
    }
    if (-not (Test-DownloadedFile $partial $ExpectedLength $ExpectedSha256)) {
        throw "Integrity check failed for downloaded file: $partial"
    }
    Move-Item -LiteralPath $partial -Destination $target -Force
    Write-Host "[dinov2:download] verified $Name"
}

if (-not $DryRun) {
    New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
}

Invoke-PinnedDownload "config.json"
Invoke-PinnedDownload "preprocessor_config.json"
Invoke-PinnedDownload "model.safetensors" $WeightLength $WeightSha256

Write-Host "DINOv2 local model ready: $OutputDirectory"
