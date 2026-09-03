[CmdletBinding()]
param(
    [string]$EnvFile = ".env",
    [string[]]$Models = @("single_topo_plain", "moco_v3_cbam"),
    [string[]]$Conditions = @(
        "mask_erode_s1", "mask_erode_s2", "mask_erode_s3",
        "mask_dilate_s1", "mask_dilate_s2", "mask_dilate_s3",
        "branch_occlusion_s1", "branch_occlusion_s2", "branch_occlusion_s3"
    ),
    [ValidateSet("all", "prepare", "extract", "evaluate")]
    [string]$Phase = "all",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExistingConfigPath = Join-Path $RepoRoot "configs\existing_models.json"
$SslConfigPath = Join-Path $RepoRoot "configs\self_supervised_models.json"
$ProtocolPath = Join-Path $RepoRoot "artifacts\manifests\query_groups.json"
$ImageRoot = Join-Path $RepoRoot "artifacts\robustness\images"
$FeatureRoot = Join-Path $RepoRoot "artifacts\robustness\features"
$ResultRoot = Join-Path $RepoRoot "artifacts\robustness\results"
$LogRoot = Join-Path $RepoRoot "artifacts\robustness\logs"
$CleanCacheRoot = Join-Path $RepoRoot "artifacts\features\validated"
$CheckpointRoot = Join-Path $RepoRoot "artifacts\checkpoints"
$ModelCache = Join-Path $RepoRoot "artifacts\models"

function Read-DotEnv {
    param([string]$Path)
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $values = @{}
    foreach ($raw in Get-Content -LiteralPath $resolved -Encoding UTF8) {
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
        if ($_ -match '[\s\[\]",]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
    }) -join " ")
}

function Invoke-Checked {
    param(
        [string]$WorkingDirectory,
        [string[]]$Arguments,
        [string]$LogPath,
        [string]$Label
    )
    Write-Host "[$Label] cwd=$WorkingDirectory"
    Write-Host "[$Label] $(Format-Command $Arguments)"
    Write-Host "[$Label] log=$LogPath"
    if ($DryRun) { return }
    New-Item -ItemType Directory -Force (Split-Path -Parent $LogPath) | Out-Null
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exitCode = 1
    Push-Location $WorkingDirectory
    try {
        & conda @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
        Pop-Location
    }
    if ($exitCode -ne 0) { throw "$Label failed with exit code $exitCode" }
}

function Normalize-List {
    param([string[]]$Values)
    return @(
        $Values |
            ForEach-Object { $_ -split "," } |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )
}

$ResolvedEnvFile = if ([IO.Path]::IsPathRooted($EnvFile)) {
    $EnvFile
} else {
    Join-Path $RepoRoot $EnvFile
}
$envValues = Read-DotEnv $ResolvedEnvFile
foreach ($key in ("MAIN_CODE_ROOT", "TEST_BINARY_ROOT")) {
    if (-not $envValues.ContainsKey($key) -or -not $envValues[$key]) {
        throw "Missing required .env key: $key"
    }
}
$MainCodeRoot = $envValues["MAIN_CODE_ROOT"]
$TestRoot = $envValues["TEST_BINARY_ROOT"]

$existingConfig = Get-Content -Raw -Encoding UTF8 $ExistingConfigPath | ConvertFrom-Json
$sslConfig = Get-Content -Raw -Encoding UTF8 $SslConfigPath | ConvertFrom-Json
$modelIndex = @{}
foreach ($model in $existingConfig.models) {
    $model | Add-Member -NotePropertyName robustness_type -NotePropertyValue "existing"
    $model | Add-Member -NotePropertyName conda_env -NotePropertyValue ([string]$existingConfig.conda_env)
    $modelIndex[[string]$model.id] = $model
}
foreach ($model in $sslConfig.models) {
    $model | Add-Member -NotePropertyName robustness_type -NotePropertyValue "ssl"
    $modelIndex[[string]$model.id] = $model
}

$Models = Normalize-List $Models
$Conditions = Normalize-List $Conditions
if ($Models.Count -eq 0) { throw "No robustness models selected" }
if ($Conditions.Count -eq 0) { throw "No robustness conditions selected" }

$selectedModels = @()
foreach ($id in $Models) {
    if (-not $modelIndex.ContainsKey($id)) { throw "Unknown robustness model id: $id" }
    $selectedModels += $modelIndex[$id]
}

$parsedConditions = @()
foreach ($condition in $Conditions) {
    if ($condition -notmatch '^(mask_erode|mask_dilate|branch_occlusion|boundary_jitter|rotation|gaussian_blur|jpeg)_s([123])$') {
        throw "Invalid robustness condition: $condition"
    }
    $parsedConditions += [PSCustomObject]@{
        id = $condition
        kind = $Matches[1]
        severity = [int]$Matches[2]
    }
}

if (-not $DryRun) {
    if (-not (Test-Path -LiteralPath $ProtocolPath -PathType Leaf)) {
        throw "Missing canonical query protocol: $ProtocolPath"
    }
    foreach ($directory in ($ImageRoot, $FeatureRoot, $ResultRoot, $LogRoot)) {
        New-Item -ItemType Directory -Force $directory | Out-Null
    }
}

if ($Phase -in @("all", "prepare")) {
    foreach ($condition in $parsedConditions) {
        $conditionImages = Join-Path $ImageRoot $condition.id
        $logPath = Join-Path $LogRoot "$($condition.id).log"
        $arguments = @(
            "run", "--no-capture-output", "-n", "gsam",
            "python", "-u", (Join-Path $RepoRoot "scripts\prepare_robustness_queries.py"),
            "--source-root", $TestRoot,
            "--query-groups", $ProtocolPath,
            "--output-root", $conditionImages,
            "--kind", $condition.kind,
            "--severity", [string]$condition.severity,
            "--seed", "42",
            "--expected-count", "1075"
        )
        Invoke-Checked $RepoRoot $arguments $logPath "prepare:$($condition.id)"
    }
}

foreach ($model in $selectedModels) {
    $id = [string]$model.id
    $cleanCache = Join-Path $CleanCacheRoot "${id}_271_1075.npz"
    if (-not $DryRun -and $Phase -in @("all", "evaluate") -and
        -not (Test-Path -LiteralPath $cleanCache -PathType Leaf)) {
        throw "Missing clean validated cache for ${id}: $cleanCache"
    }

    foreach ($condition in $parsedConditions) {
        $conditionImages = Join-Path $ImageRoot $condition.id
        $shiftedCache = Join-Path $FeatureRoot "${id}__$($condition.id).pt"
        $resultJson = Join-Path $ResultRoot "${id}__$($condition.id).json"
        $resultCsv = Join-Path $ResultRoot "${id}__$($condition.id)_per_query.csv"
        $logPath = Join-Path $LogRoot "${id}__$($condition.id).log"

        if ($Phase -in @("all", "extract")) {
            if ($model.robustness_type -eq "existing") {
                $workdir = Join-Path $MainCodeRoot ([string]$model.workdir)
                $checkpoint = Join-Path $MainCodeRoot ([string]$model.checkpoint)
                $arguments = @(
                    "run", "--no-capture-output", "-n", [string]$model.conda_env,
                    "python", "-u", [string]$model.extractor.script
                )
                if ($model.extractor.mode -eq "argparse") {
                    $arguments += @(
                        "--image-dir", $conditionImages,
                        "--model-path", $checkpoint,
                        "--output-feats", $shiftedCache
                    )
                    $arguments += @($model.extractor.arguments | ForEach-Object { [string]$_ })
                }
                elseif ($model.extractor.mode -eq "override") {
                    $arguments += @(
                        "--override", "image_dir=$conditionImages",
                        "--override", "model_path=$checkpoint",
                        "--override", "output_feats=$shiftedCache"
                    )
                    foreach ($override in $model.extractor.arguments) {
                        $arguments += @("--override", [string]$override)
                    }
                }
                else {
                    throw "Unsupported extractor mode for ${id}: $($model.extractor.mode)"
                }
                Invoke-Checked $workdir $arguments $logPath "${id}:$($condition.id):extract"
            }
            elseif ($model.robustness_type -eq "ssl") {
                $checkpoint = Join-Path (Join-Path $CheckpointRoot $id) "best.pt"
                if (-not $DryRun -and -not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
                    throw "Missing self-supervised checkpoint for ${id}: $checkpoint"
                }
                $arguments = @(
                    "run", "--no-capture-output", "-n", [string]$model.conda_env,
                    "python", "-u", (Join-Path $RepoRoot "scripts\extract_ssl_checkpoint.py"),
                    "--config", $SslConfigPath,
                    "--model-id", $id,
                    "--checkpoint", $checkpoint,
                    "--image-dir", $conditionImages,
                    "--output", $shiftedCache,
                    "--model-cache", $ModelCache,
                    "--expected-count", "1075",
                    "--batch-size", [string]$model.batch_size,
                    "--num-workers", "4",
                    "--device", "auto"
                )
                Invoke-Checked $RepoRoot $arguments $logPath "${id}:$($condition.id):extract"
            }
        }

        if ($Phase -in @("all", "evaluate")) {
            if (-not $DryRun -and -not (Test-Path -LiteralPath $shiftedCache -PathType Leaf)) {
                throw "Missing shifted query feature cache: $shiftedCache"
            }
            $arguments = @(
                "run", "--no-capture-output", "-n", "gsam",
                "python", "-u", (Join-Path $RepoRoot "scripts\evaluate_query_shift.py"),
                "--clean-cache", $cleanCache,
                "--shifted-cache", $shiftedCache,
                "--query-groups", $ProtocolPath,
                "--condition", $condition.id,
                "--output", $resultJson,
                "--per-query-csv", $resultCsv,
                "--ks", "1,5,10,20",
                "--expected-query-count", "1075",
                "--bootstrap-iterations", "2000",
                "--bootstrap-seed", "42",
                "--trusted-local-pt"
            )
            Invoke-Checked $RepoRoot $arguments $logPath "${id}:$($condition.id):evaluate"
        }
    }
}

Write-Host "Completed phase=$Phase models=$($Models -join ',') conditions=$($Conditions -join ',') dry_run=$([bool]$DryRun)"
