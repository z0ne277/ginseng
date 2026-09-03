[CmdletBinding()]
param(
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CheckScript = Join-Path $RepoRoot "scripts\check_gsam_dinov2_env.py"

function Format-Command {
    param([string[]]$Arguments)
    return "conda " + (($Arguments | ForEach-Object {
        if ($_ -match '[\s,]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join " ")
}

function Invoke-Checked {
    param([string[]]$Arguments, [string]$Label)
    Write-Host "[$Label] $(Format-Command $Arguments)"
    if ($DryRun) { return }
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & conda @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) { throw "$Label failed with exit code $exitCode" }
}

Invoke-Checked @(
    "run", "--no-capture-output", "-n", "gsam",
    "python", "-m", "pip", "install",
    "transformers==4.35.2", "tokenizers==0.15.2", "--no-deps"
) "install-dinov2-transformers"

Invoke-Checked @(
    "run", "--no-capture-output", "-n", "gsam",
    "python", "-u", $CheckScript
) "check-gsam"

Write-Host "gsam DINOv2 setup completed dry_run=$([bool]$DryRun)"
