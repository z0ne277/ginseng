[CmdletBinding()]
param(
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvironmentFile = Join-Path $RepoRoot "environment-modern.yml"
$RequirementsFile = Join-Path $RepoRoot "requirements-modern.txt"
$CheckScript = Join-Path $RepoRoot "scripts\check_modern_env.py"
$EnvironmentName = "ginseng-baselines"

function Format-Command {
    param([string]$Executable, [string[]]$Arguments)
    return $Executable + " " + (($Arguments | ForEach-Object {
        if ($_ -match '[\s,]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join " ")
}

function Invoke-SetupStep {
    param([string]$Executable, [string[]]$Arguments, [string]$Label)
    Write-Host "[$Label] $(Format-Command $Executable $Arguments)"
    if ($DryRun) { return }
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Executable @Arguments
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) { throw "$Label failed with exit code $exitCode" }
}

function Find-EnvironmentPrefix {
    $environmentList = (& conda env list --json | ConvertFrom-Json).envs
    $matches = @($environmentList | Where-Object {
        (Split-Path -Leaf $_) -eq $EnvironmentName
    })
    if ($matches.Count -gt 1) { throw "Multiple conda environments named $EnvironmentName" }
    if ($matches.Count -eq 1) { return [string]$matches[0] }
    return ""
}

$EnvironmentPrefix = if ($DryRun) { "" } else { Find-EnvironmentPrefix }
if ($DryRun -or -not $EnvironmentPrefix) {
    Invoke-SetupStep "conda" @("env", "create", "-f", $EnvironmentFile) "create-env"
    if (-not $DryRun) {
        $EnvironmentPrefix = Find-EnvironmentPrefix
        if (-not $EnvironmentPrefix) { throw "Conda environment was not created: $EnvironmentName" }
    }
}
else {
    Write-Host "[create-env] $EnvironmentName already exists; preserving it"
}

$EnvironmentPrefix = if ($DryRun) {
    Join-Path "<conda-envs>" $EnvironmentName
} else { $EnvironmentPrefix }
$EnvironmentPython = Join-Path $EnvironmentPrefix "python.exe"
$PipTempRoot = Join-Path $RepoRoot "artifacts\tmp\pip-modern-$PID"
$PipCacheRoot = Join-Path $RepoRoot "artifacts\pip-cache"
$PipNetworkArguments = @(
    "--timeout", "600",
    "--retries", "20",
    "--resume-retries", "20",
    "--cache-dir", $PipCacheRoot,
    "--disable-pip-version-check"
)

$previousTemp = $env:TEMP
$previousTmp = $env:TMP
try {
    if (-not $DryRun) {
        New-Item -ItemType Directory -Path $PipTempRoot, $PipCacheRoot -Force | Out-Null
        $env:TEMP = $PipTempRoot
        $env:TMP = $PipTempRoot
    }

    Invoke-SetupStep $EnvironmentPython (@(
        "-m", "pip", "install",
        "torch==2.7.1", "torchvision==0.22.1", "torchaudio==2.7.1",
        "--index-url", "https://download.pytorch.org/whl/cu126"
    ) + $PipNetworkArguments) "install-pytorch"

    Invoke-SetupStep $EnvironmentPython (@(
        "-m", "pip", "install", "-r", $RequirementsFile
    ) + $PipNetworkArguments) "install-modern-requirements"

    Invoke-SetupStep $EnvironmentPython (@(
        "-m", "pip", "install", "-e", $RepoRoot, "--no-deps"
    ) + $PipNetworkArguments) "install-project"

    Invoke-SetupStep $EnvironmentPython @("-u", $CheckScript) "check-environment"
}
finally {
    $env:TEMP = $previousTemp
    $env:TMP = $previousTmp
}

Write-Host "Modern environment setup completed dry_run=$([bool]$DryRun)"
