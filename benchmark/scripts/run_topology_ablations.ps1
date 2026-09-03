[CmdletBinding()]
param(
    [string]$EnvFile = ".env",
    [string[]]$Variants = @(),
    [ValidateSet("all", "train", "extract", "stamp", "evaluate")]
    [string]$Phase = "all",
    [int]$Seed = 42,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$MatrixPath = Join-Path $RepoRoot "configs\topology_training_ablations.json"
$ProtocolPath = Join-Path $RepoRoot "artifacts\manifests\query_groups.json"
$CheckpointRoot = Join-Path $RepoRoot "artifacts\checkpoints\topology_ablations"
$RawRoot = Join-Path $RepoRoot "artifacts\features\raw"
$ValidatedRoot = Join-Path $RepoRoot "artifacts\features\validated"
$ResultRoot = Join-Path $RepoRoot "artifacts\results"
$LogRoot = Join-Path $RepoRoot "artifacts\logs\topology_ablations"

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

function ConvertTo-UrlSafeBase64 {
    param([string]$Json)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Json))
    return $encoded.TrimEnd([char[]]"=").Replace("+", "-").Replace("/", "_")
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

function Format-OverrideValue {
    param($Value)
    if ($Value -is [bool]) {
        if ($Value) { return "true" }
        return "false"
    }
    return [string]$Value
}

function Add-Overrides {
    param([string[]]$Arguments, $Overrides)
    $result = @($Arguments)
    foreach ($property in $Overrides.PSObject.Properties) {
        $result += @(
            "--override",
            "$($property.Name)=$(Format-OverrideValue $property.Value)"
        )
    }
    return $result
}

$ResolvedEnvFile = if ([IO.Path]::IsPathRooted($EnvFile)) {
    $EnvFile
} else {
    Join-Path $RepoRoot $EnvFile
}
$envValues = Read-DotEnv $ResolvedEnvFile
foreach ($key in ("MAIN_CODE_ROOT", "MERGED_GALLERY")) {
    if (-not $envValues.ContainsKey($key) -or -not $envValues[$key]) {
        throw "Missing required .env key: $key"
    }
}
$MainCodeRoot = $envValues["MAIN_CODE_ROOT"]
$GalleryRoot = $envValues["MERGED_GALLERY"]
$WorkDir = Join-Path $MainCodeRoot "single_topo"

$TrainCsv = if ($envValues.ContainsKey("TRAIN_CSV") -and $envValues["TRAIN_CSV"]) {
    $envValues["TRAIN_CSV"]
} else {
    Join-Path $WorkDir "csv\train.csv"
}
$ValCsv = if ($envValues.ContainsKey("VAL_CSV") -and $envValues["VAL_CSV"]) {
    $envValues["VAL_CSV"]
} else {
    Join-Path $WorkDir "csv\val.csv"
}
$DevCsv = if ($envValues.ContainsKey("DEV_CSV") -and $envValues["DEV_CSV"]) {
    $envValues["DEV_CSV"]
} else {
    Join-Path $WorkDir "csv\test.csv"
}

$matrix = Get-Content -Raw -Encoding UTF8 $MatrixPath | ConvertFrom-Json
if ($matrix.schema_version -ne 1 -or $matrix.protocol_tag -ne "271_1075_unlabeled") {
    throw "Unsupported topology ablation matrix"
}
$selected = @($matrix.variants)
$Variants = Normalize-List $Variants
if ($Variants.Count -gt 0) {
    $requested = @{}
    foreach ($id in $Variants) { $requested[$id.ToLowerInvariant()] = $true }
    $selected = @(
        $selected |
            Where-Object { $requested.ContainsKey($_.id.ToLowerInvariant()) }
    )
    $found = @($selected | ForEach-Object { $_.id.ToLowerInvariant() })
    $missing = @($requested.Keys | Where-Object { $_ -notin $found })
    if ($missing.Count -gt 0) { throw "Unknown ablation variant(s): $($missing -join ', ')" }
}
if ($selected.Count -eq 0) { throw "No topology ablation variants selected" }
if ($Seed -lt 0) { throw "Seed must be non-negative" }

foreach ($variant in $selected) {
    if ([bool]$variant.requires_identity_labels) {
        throw "Identity-supervised variant is forbidden by the current image-only protocol"
    }
    $id = [string]$variant.id
    $seedSuffix = if ($Seed -eq 42) { "" } else { "_seed${Seed}" }
    $artifactId = "topology_ablation_${id}${seedSuffix}"
    $checkpointDir = Join-Path $CheckpointRoot "${id}${seedSuffix}"
    $checkpoint = Join-Path $checkpointDir "best_model.pth"
    $rawCache = Join-Path $RawRoot "${artifactId}_271_1075.pt"
    $validatedCache = Join-Path $ValidatedRoot "${artifactId}_271_1075.npz"
    $resultJson = Join-Path $ResultRoot "${artifactId}_271_1075.json"
    $resultCsv = Join-Path $ResultRoot "${artifactId}_271_1075_per_query.csv"
    $logPath = Join-Path $LogRoot "${id}${seedSuffix}.log"
    $featureDim = 256 + [int]$variant.overrides.topo_dim

    if ($Phase -in @("all", "train")) {
        $arguments = @(
            "run", "--no-capture-output", "-n", "gsam",
            "python", "-u", "train.py"
        )
        $arguments = Add-Overrides $arguments $variant.overrides
        $arguments += @(
            "--override", "train_csv=$TrainCsv",
            "--override", "val_csv=$ValCsv",
            "--override", "test_csv=$DevCsv",
            "--override", "checkpoint_dir=$checkpointDir",
            "--override", "seed=$Seed",
            "--override", "tta_enabled=false"
        )
        Invoke-Checked $WorkDir $arguments $logPath "${id}:train"
    }

    if ($Phase -in @("all", "extract")) {
        if (-not $DryRun -and -not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "Missing trained ablation checkpoint: $checkpoint"
        }
        $arguments = @(
            "run", "--no-capture-output", "-n", "gsam",
            "python", "-u", "extraction.py"
        )
        $arguments = Add-Overrides $arguments $variant.overrides
        $arguments += @(
            "--override", "pretrained_backbone=false",
            "--override", "image_dir=$GalleryRoot",
            "--override", "model_path=$checkpoint",
            "--override", "output_feats=$rawCache",
            "--override", "feature_type=both",
            "--override", "tta_enabled=false",
            "--override", "tta_modes=stretch224",
            "--override", "tta_weights=[1.0]"
        )
        Invoke-Checked $WorkDir $arguments $logPath "${id}:extract"
    }

    if ($Phase -in @("all", "stamp")) {
        $preprocessing = @{
            source = "same project foreground images"
            resize = "stretch224"
            training_information = "image-only self-supervision; no identity labels"
        } | ConvertTo-Json -Compress
        $tta = @{ enabled = $false; weights = @(1.0) } | ConvertTo-Json -Compress
        $environment = @{
            conda_env = "gsam"
            ablation_group = [string]$variant.group
            seed = $Seed
        } | ConvertTo-Json -Compress
        $arguments = @(
            "run", "--no-capture-output", "-n", "gsam",
            "python", "-u", (Join-Path $RepoRoot "scripts\stamp_feature_cache.py"),
            "--env", $ResolvedEnvFile,
            "--raw-cache", $rawCache,
            "--output", $validatedCache,
            "--model-id", $artifactId,
            "--model-source", "task-trained:controlled-topology-ablation",
            "--checkpoint", $checkpoint,
            "--feature-normalization", "l2",
            "--preprocessing-json-base64", (ConvertTo-UrlSafeBase64 $preprocessing),
            "--tta-json-base64", (ConvertTo-UrlSafeBase64 $tta),
            "--environment-json-base64", (ConvertTo-UrlSafeBase64 $environment),
            "--expected-feature-dim", [string]$featureDim,
            "--trusted-local-pt"
        )
        Invoke-Checked $RepoRoot $arguments $logPath "${id}:stamp"
    }

    if ($Phase -in @("all", "evaluate")) {
        $arguments = @(
            "run", "--no-capture-output", "-n", "gsam",
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
        Invoke-Checked $RepoRoot $arguments $logPath "${id}:evaluate"
    }
}

Write-Host "Completed phase=$Phase variants=$($selected.id -join ',') seed=$Seed dry_run=$([bool]$DryRun)"
