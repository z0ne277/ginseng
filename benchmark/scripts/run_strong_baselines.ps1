[CmdletBinding()]
param(
    [string]$EnvFile = ".env",
    [string[]]$Models = @(),
    [ValidateSet("all", "extract", "stamp", "evaluate")]
    [string]$Phase = "all",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigPath = Join-Path $RepoRoot "configs\strong_models.json"
$ProtocolPath = Join-Path $RepoRoot "artifacts\manifests\query_groups.json"
$RawRoot = Join-Path $RepoRoot "artifacts\features\raw"
$ValidatedRoot = Join-Path $RepoRoot "artifacts\features\validated"
$ResultRoot = Join-Path $RepoRoot "artifacts\results"
$LogRoot = Join-Path $RepoRoot "artifacts\logs"

function Read-DotEnv {
    param([string]$Path)
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $values = @{}
    $lineNumber = 0
    foreach ($raw in Get-Content -LiteralPath $resolved -Encoding UTF8) {
        $lineNumber++
        $line = $raw.Trim()
        if (-not $line -or $line.StartsWith("#")) { continue }
        $separator = $line.IndexOf("=")
        if ($separator -lt 1) { throw "Invalid .env entry at line $lineNumber" }
        $key = $line.Substring(0, $separator).Trim()
        $value = $line.Substring($separator + 1).Trim()
        if (($value.StartsWith('"') -and $value.EndsWith('"')) -or
            ($value.StartsWith("'") -and $value.EndsWith("'"))) {
            $value = $value.Substring(1, $value.Length - 2)
        }
        $values[$key] = $value
    }
    return $values
}

function Require-Directory {
    param([hashtable]$Values, [string]$Key)
    if (-not $Values.ContainsKey($Key) -or -not $Values[$Key]) {
        throw "Missing required .env key: $Key"
    }
    return (Get-Item -LiteralPath $Values[$Key] -ErrorAction Stop).FullName
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

function Test-CompleteHfModelDirectory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }
    foreach ($name in ("config.json", "preprocessor_config.json", "model.safetensors")) {
        $file = Get-Item -LiteralPath (Join-Path $Path $name) -ErrorAction SilentlyContinue
        if (-not $file -or $file.PSIsContainer -or $file.Length -le 0) { return $false }
    }
    return $true
}

function Invoke-CheckedCommand {
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
    New-Item -ItemType Directory -Path (Split-Path -Parent $LogPath) -Force | Out-Null
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exitCode = 1
    Push-Location $WorkingDirectory
    try {
        & conda @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Pop-Location
    }
    if ($exitCode -ne 0) { throw "$Label failed with exit code $exitCode" }
}

$resolvedEnvFile = if ([IO.Path]::IsPathRooted($EnvFile)) {
    $EnvFile
} else {
    Join-Path $RepoRoot $EnvFile
}
$envValues = Read-DotEnv $resolvedEnvFile
$GalleryRoot = Require-Directory $envValues "MERGED_GALLERY"
$config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($config.schema_version -ne 1 -or $config.protocol_tag -ne "271_1075") {
    throw "Unsupported strong-model configuration"
}
if (-not $DryRun -and -not (Test-Path -LiteralPath $ProtocolPath -PathType Leaf)) {
    throw "Missing canonical query protocol: $ProtocolPath. Run build_query_groups.py first."
}

$selected = @($config.models)
if ($Models.Count -gt 0) {
    $Models = @(
        $Models |
            ForEach-Object { $_ -split "," } |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ }
    )
    $requested = @{}
    foreach ($name in $Models) { $requested[$name.ToLowerInvariant()] = $true }
    $selected = @($selected | Where-Object { $requested.ContainsKey($_.id.ToLowerInvariant()) })
    $found = @($selected | ForEach-Object { $_.id.ToLowerInvariant() })
    $missing = @($requested.Keys | Where-Object { $_ -notin $found })
    if ($missing.Count -gt 0) { throw "Unknown model id(s): $($missing -join ', ')" }
}
if ($selected.Count -eq 0) { throw "No models selected" }

if (-not $DryRun) {
    foreach ($directory in ($RawRoot, $ValidatedRoot, $ResultRoot, $LogRoot)) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
}

$doExtract = $Phase -in @("all", "extract")
$doStamp = $Phase -in @("all", "stamp")
$doEvaluate = $Phase -in @("all", "evaluate")

foreach ($model in $selected) {
    $id = [string]$model.id
    $condaEnv = [string]$model.conda_env
    $loadModel = [string]$model.model
    $usingLocalModel = $false
    $localModelEnvKey = if ($model.PSObject.Properties.Name -contains "local_model_env_key") {
        [string]$model.local_model_env_key
    } else { "" }
    if ($localModelEnvKey -and $envValues.ContainsKey($localModelEnvKey) -and $envValues[$localModelEnvKey]) {
        $localModel = Get-Item -LiteralPath $envValues[$localModelEnvKey] -ErrorAction Stop
        if (-not $localModel.PSIsContainer -or -not (Test-CompleteHfModelDirectory $localModel.FullName)) {
            throw "$localModelEnvKey must point to a complete Hugging Face model directory"
        }
        $loadModel = $localModel.FullName
        $usingLocalModel = $true
    }
    $defaultLocalModelDir = if ($model.PSObject.Properties.Name -contains "default_local_model_dir") {
        Join-Path $RepoRoot ([string]$model.default_local_model_dir)
    } else { "" }
    if (-not $usingLocalModel -and $defaultLocalModelDir -and
        (Test-CompleteHfModelDirectory $defaultLocalModelDir)) {
        $loadModel = (Resolve-Path -LiteralPath $defaultLocalModelDir).Path
        $usingLocalModel = $true
    }
    $prefetchScript = if ($model.PSObject.Properties.Name -contains "prefetch_script") {
        Join-Path $RepoRoot ([string]$model.prefetch_script)
    } else { "" }
    if ($doExtract -and -not $usingLocalModel -and $prefetchScript) {
        Write-Host "[${id}:prefetch] powershell -NoProfile -ExecutionPolicy Bypass -File $prefetchScript -OutputDirectory $defaultLocalModelDir"
        if (-not $DryRun) {
            & powershell -NoProfile -ExecutionPolicy Bypass -File $prefetchScript -OutputDirectory $defaultLocalModelDir
            if ($LASTEXITCODE -ne 0) { throw "${id}:prefetch failed with exit code $LASTEXITCODE" }
            if (-not (Test-CompleteHfModelDirectory $defaultLocalModelDir)) {
                throw "${id}:prefetch did not create a complete model directory: $defaultLocalModelDir"
            }
            $loadModel = (Resolve-Path -LiteralPath $defaultLocalModelDir).Path
            $usingLocalModel = $true
        }
    }
    $isGated = ($model.PSObject.Properties.Name -contains "gated") -and [bool]$model.gated
    $hubToken = if ($envValues.ContainsKey("HF_TOKEN")) { [string]$envValues["HF_TOKEN"] } else { "" }
    if ($doExtract -and -not $DryRun -and $isGated -and -not $usingLocalModel -and -not $hubToken.Trim()) {
        throw (
            "$id is a gated Hugging Face model. Accept its license at " +
            "https://huggingface.co/$($model.model), then set HF_TOKEN in .env; " +
            "alternatively set $localModelEnvKey to a complete local model directory."
        )
    }
    $rawCache = Join-Path $RawRoot "${id}_271_1075.pt"
    $validatedCache = Join-Path $ValidatedRoot "${id}_271_1075.npz"
    $resultJson = Join-Path $ResultRoot "${id}_271_1075.json"
    $resultCsv = Join-Path $ResultRoot "${id}_271_1075_per_query.csv"
    $logPath = Join-Path $LogRoot "${id}_271_1075.log"

    if ($doExtract) {
        $extractorKind = if ($model.PSObject.Properties.Name -contains "extractor_kind") {
            [string]$model.extractor_kind
        } else { "pooler_or_cls" }
        $extractArguments = @(
            "run", "--no-capture-output", "-n", $condaEnv,
            "python", "-u",
            (Join-Path $RepoRoot "scripts\extract_hf_vision.py"),
            "--env", $resolvedEnvFile,
            "--image-dir", $GalleryRoot,
            "--output", $rawCache,
            "--model", $loadModel,
            "--revision", [string]$model.revision,
            "--extractor-kind", $extractorKind,
            "--expected-dim", [string]$model.feature_dim,
            "--expected-count", "12787",
            "--batch-size", [string]$model.batch_size,
            "--num-workers", [string]$model.num_workers,
            "--device", "auto",
            "--token-env-key", "HF_TOKEN"
        )
        Invoke-CheckedCommand $RepoRoot $extractArguments $logPath "${id}:extract"
    }

    if ($doStamp) {
        if (-not $DryRun -and -not (Test-Path -LiteralPath $rawCache -PathType Leaf)) {
            throw "Missing raw cache for ${id}: $rawCache"
        }
        $preprocessingJson = $model.preprocessing | ConvertTo-Json -Compress -Depth 10
        $ttaJson = $model.tta | ConvertTo-Json -Compress -Depth 10
        $environmentJson = @{
            conda_env = $condaEnv
            model_revision = [string]$model.revision
        } | ConvertTo-Json -Compress
        $preprocessingJson = ConvertTo-UrlSafeBase64 $preprocessingJson
        $ttaJson = ConvertTo-UrlSafeBase64 $ttaJson
        $environmentJson = ConvertTo-UrlSafeBase64 $environmentJson
        $modelSource = "huggingface:$($model.model)@$($model.revision)"
        $stampArguments = @(
            "run", "--no-capture-output", "-n", $condaEnv,
            "python", "-u",
            (Join-Path $RepoRoot "scripts\stamp_feature_cache.py"),
            "--env", $resolvedEnvFile,
            "--raw-cache", $rawCache,
            "--output", $validatedCache,
            "--model-id", $id,
            "--model-source", $modelSource,
            "--feature-normalization", "l2",
            "--preprocessing-json-base64", $preprocessingJson,
            "--tta-json-base64", $ttaJson,
            "--environment-json-base64", $environmentJson,
            "--expected-feature-dim", [string]$model.feature_dim,
            "--trusted-local-pt"
        )
        Invoke-CheckedCommand $RepoRoot $stampArguments $logPath "${id}:stamp"
    }

    if ($doEvaluate) {
        if (-not $DryRun -and -not (Test-Path -LiteralPath $validatedCache -PathType Leaf)) {
            throw "Missing validated cache for ${id}: $validatedCache"
        }
        $evaluateArguments = @(
            "run", "--no-capture-output", "-n", $condaEnv,
            "python", "-u",
            (Join-Path $RepoRoot "scripts\evaluate_features.py"),
            "--cache", $validatedCache,
            "--query-groups", $ProtocolPath,
            "--output", $resultJson,
            "--per-query-csv", $resultCsv,
            "--ks", "1,5,10,20",
            "--block-size", "32",
            "--bootstrap-iterations", "2000",
            "--bootstrap-seed", "42"
        )
        Invoke-CheckedCommand $RepoRoot $evaluateArguments $logPath "${id}:evaluate"
    }
}

Write-Host "Completed phase=$Phase models=$($selected.id -join ',') dry_run=$([bool]$DryRun)"
