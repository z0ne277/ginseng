[CmdletBinding()]
param(
    [string]$EnvFile = ".env",
    [string[]]$Models = @(),
    [ValidateSet("all", "train", "extract", "stamp", "evaluate")]
    [string]$Phase = "all",
    [int]$Seed = 42,
    [switch]$Resume,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigPath = Join-Path $RepoRoot "configs\self_supervised_models.json"
$ProtocolPath = Join-Path $RepoRoot "artifacts\manifests\query_groups.json"
$CheckpointRoot = Join-Path $RepoRoot "artifacts\checkpoints"
$ModelCache = Join-Path $RepoRoot "artifacts\models"
$RawRoot = Join-Path $RepoRoot "artifacts\features\raw"
$ValidatedRoot = Join-Path $RepoRoot "artifacts\features\validated"
$ResultRoot = Join-Path $RepoRoot "artifacts\results"
$LogRoot = Join-Path $RepoRoot "artifacts\logs"

function Read-DotEnv {
    param([string]$Path)
    $values = @{}
    foreach ($raw in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }
        $separator = $line.IndexOf("=")
        if ($separator -lt 1) { throw "Invalid .env entry: $raw" }
        $key = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim().Trim('"').Trim("'")
        $values[$key] = $value
    }
    return $values
}

function Format-Command {
    param([string[]]$Arguments)
    return "conda " + (($Arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }) -join " ")
}

function ConvertTo-UrlSafeBase64 {
    param([string]$Json)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Json))
    return $encoded.TrimEnd([char[]]"=").Replace("+", "-").Replace("/", "_")
}

function Invoke-Checked {
    param(
        [string[]]$Arguments,
        [string]$LogPath,
        [string]$Label
    )
    Write-Host "[$Label] $(Format-Command $Arguments)"
    Write-Host "[$Label] log=$LogPath"
    if ($DryRun) { return }
    New-Item -ItemType Directory -Force (Split-Path -Parent $LogPath) | Out-Null
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & conda @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $oldPreference
    }
    if ($exitCode -ne 0) { throw "$Label failed with exit code $exitCode" }
}

$ResolvedEnvFile = if ([IO.Path]::IsPathRooted($EnvFile)) {
    $EnvFile
} else {
    Join-Path $RepoRoot $EnvFile
}
$envValues = Read-DotEnv $ResolvedEnvFile
$mainCodeRoot = if ($envValues.ContainsKey("MAIN_CODE_ROOT")) {
    $envValues["MAIN_CODE_ROOT"]
} else {
    (Split-Path -Parent $RepoRoot)
}
$TrainCsv = if ($envValues.ContainsKey("TRAIN_CSV") -and $envValues["TRAIN_CSV"]) {
    $envValues["TRAIN_CSV"]
} else {
    Join-Path $mainCodeRoot "data\splits\train.csv"
}
$ValCsv = if ($envValues.ContainsKey("VAL_CSV") -and $envValues["VAL_CSV"]) {
    $envValues["VAL_CSV"]
} else {
    Join-Path $mainCodeRoot "data\splits\val.csv"
}
$DevCsv = if ($envValues.ContainsKey("DEV_CSV") -and $envValues["DEV_CSV"]) {
    $envValues["DEV_CSV"]
} else {
    Join-Path $mainCodeRoot "data\splits\test.csv"
}
if (-not $envValues.ContainsKey("MERGED_GALLERY") -or -not $envValues["MERGED_GALLERY"]) {
    throw "MERGED_GALLERY is required in .env"
}
$GalleryRoot = $envValues["MERGED_GALLERY"]

$config = Get-Content -Raw -Encoding UTF8 $ConfigPath | ConvertFrom-Json
if ($config.schema_version -ne 1 -or $config.protocol_tag -ne "271_1075_unlabeled") {
    throw "Unsupported self-supervised configuration"
}
$selected = @($config.models)
if ($Models.Count -gt 0) {
    $requested = @{}
    foreach ($value in $Models) {
        foreach ($name in ($value -split ",")) {
            if ($name.Trim()) { $requested[$name.Trim().ToLowerInvariant()] = $true }
        }
    }
    $selected = @($selected | Where-Object { $requested.ContainsKey($_.id.ToLowerInvariant()) })
    $found = @($selected | ForEach-Object { $_.id.ToLowerInvariant() })
    $missing = @($requested.Keys | Where-Object { $_ -notin $found })
    if ($missing.Count -gt 0) { throw "Unknown model id(s): $($missing -join ', ')" }
}
if ($selected.Count -eq 0) { throw "No models selected" }
if ($Seed -lt 0) { throw "Seed must be non-negative" }

foreach ($model in $selected) {
    $id = [string]$model.id
    $seedSuffix = if ($Seed -eq 42) { "" } else { "_seed${Seed}" }
    $artifactId = "${id}${seedSuffix}"
    $condaEnv = [string]$model.conda_env
    $checkpointDir = Join-Path $CheckpointRoot $artifactId
    $checkpoint = Join-Path $checkpointDir "best.pt"
    $rawCache = Join-Path $RawRoot "${artifactId}_271_1075.pt"
    $validatedCache = Join-Path $ValidatedRoot "${artifactId}_271_1075.npz"
    $resultJson = Join-Path $ResultRoot "${artifactId}_271_1075.json"
    $resultCsv = Join-Path $ResultRoot "${artifactId}_271_1075_per_query.csv"
    $logPath = Join-Path $LogRoot "${artifactId}_271_1075.log"

    if ($Phase -in @("all", "train")) {
        $arguments = @(
            "run", "--no-capture-output", "-n", $condaEnv,
            "python", "-u", (Join-Path $RepoRoot "scripts\train_ssl_baseline.py"),
            "--config", $ConfigPath,
            "--model-id", $id,
            "--train-csv", $TrainCsv,
            "--val-csv", $ValCsv,
            "--dev-csv", $DevCsv,
            "--output-dir", $checkpointDir,
            "--model-cache", $ModelCache,
            "--device", "auto",
            "--num-workers", "4",
            "--seed", [string]$Seed
        )
        if ($Resume) { $arguments += "--resume" }
        Invoke-Checked $arguments $logPath "${id}:train"
    }

    if ($Phase -in @("all", "extract")) {
        if (-not $DryRun -and -not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "Missing checkpoint: $checkpoint"
        }
        $arguments = @(
            "run", "--no-capture-output", "-n", $condaEnv,
            "python", "-u", (Join-Path $RepoRoot "scripts\extract_ssl_checkpoint.py"),
            "--config", $ConfigPath,
            "--model-id", $artifactId,
            "--checkpoint", $checkpoint,
            "--image-dir", $GalleryRoot,
            "--output", $rawCache,
            "--model-cache", $ModelCache,
            "--expected-count", "12787",
            "--batch-size", [string]$model.batch_size,
            "--num-workers", "4",
            "--device", "auto"
        )
        Invoke-Checked $arguments $logPath "${id}:extract"
    }

    if ($Phase -in @("all", "stamp")) {
        $preprocessing = @{
            source = "project foreground images"
            resize = 256
            crop = 224
            normalization = "ImageNet"
            training_information = "image-only self-supervision; no identity labels"
        } | ConvertTo-Json -Compress
        $environment = @{
            conda_env = $condaEnv
            seed = $Seed
        } | ConvertTo-Json -Compress
        $arguments = @(
            "run", "--no-capture-output", "-n", $condaEnv,
            "python", "-u", (Join-Path $RepoRoot "scripts\stamp_feature_cache.py"),
            "--env", $ResolvedEnvFile,
            "--raw-cache", $rawCache,
            "--output", $validatedCache,
            "--model-id", $id,
            "--model-source", "task-trained:$($model.algorithm):$($model.backbone)",
            "--checkpoint", $checkpoint,
            "--feature-normalization", "l2",
            "--preprocessing-json-base64", (ConvertTo-UrlSafeBase64 $preprocessing),
            "--tta-json-base64", (ConvertTo-UrlSafeBase64 '{"enabled":false,"weights":[1.0]}'),
            "--environment-json-base64", (ConvertTo-UrlSafeBase64 $environment),
            "--expected-feature-dim", [string]$model.feature_dim,
            "--trusted-local-pt"
        )
        Invoke-Checked $arguments $logPath "${id}:stamp"
    }

    if ($Phase -in @("all", "evaluate")) {
        $arguments = @(
            "run", "--no-capture-output", "-n", $condaEnv,
            "python", "-u", (Join-Path $RepoRoot "scripts\evaluate_features.py"),
            "--cache", $validatedCache,
            "--query-groups", $ProtocolPath,
            "--output", $resultJson,
            "--per-query-csv", $resultCsv,
            "--ks", "1,5,10,20",
            "--block-size", "32",
            "--bootstrap-iterations", "2000",
            "--bootstrap-seed", "42"
        )
        Invoke-Checked $arguments $logPath "${id}:evaluate"
    }
}

Write-Host "Completed phase=$Phase models=$($selected.id -join ',') seed=$Seed dry_run=$([bool]$DryRun)"
