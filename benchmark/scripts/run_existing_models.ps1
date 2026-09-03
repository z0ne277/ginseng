[CmdletBinding()]
param(
    [string]$EnvFile = ".env",
    [Alias("Config")]
    [string]$ModelConfig = "configs\existing_models.json",
    [string[]]$Models = @(),
    [ValidateSet("all", "extract", "stamp", "evaluate")]
    [string]$Phase = "all",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigPath = if ([IO.Path]::IsPathRooted($ModelConfig)) {
    $ModelConfig
} else {
    Join-Path $RepoRoot $ModelConfig
}
$ConfigPath = (Resolve-Path -LiteralPath $ConfigPath).Path
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

function ConvertTo-NativeJsonArgument {
    param([string]$Json)
    return $Json -replace '"', '\"'
}

function Invoke-CheckedCommand {
    param(
        [string]$WorkingDirectory,
        [string[]]$Arguments,
        [string]$LogPath,
        [string]$Label
    )
    $display = Format-Command $Arguments
    Write-Host "[$Label] cwd=$WorkingDirectory"
    Write-Host "[$Label] $display"
    Write-Host "[$Label] log=$LogPath"
    if ($DryRun) { return }
    $logParent = Split-Path -Parent $LogPath
    New-Item -ItemType Directory -Path $logParent -Force | Out-Null
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
$MainCodeRoot = Require-Directory $envValues "MAIN_CODE_ROOT"
$GalleryRoot = Require-Directory $envValues "MERGED_GALLERY"
$config = Get-Content -LiteralPath $ConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($config.schema_version -ne 1 -or $config.protocol_tag -ne "271_1075") {
    throw "Unsupported model configuration: $ConfigPath"
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
    $workdir = (Get-Item -LiteralPath (Join-Path $MainCodeRoot $model.workdir)).FullName
    $checkpoint = (Get-Item -LiteralPath (Join-Path $MainCodeRoot $model.checkpoint)).FullName
    $rawCache = Join-Path $RawRoot "${id}_271_1075.pt"
    $validatedCache = Join-Path $ValidatedRoot "${id}_271_1075.npz"
    $resultJson = Join-Path $ResultRoot "${id}_271_1075.json"
    $resultCsv = Join-Path $ResultRoot "${id}_271_1075_per_query.csv"
    $logPath = Join-Path $LogRoot "${id}_271_1075.log"

    if ($doExtract) {
        $extractArguments = @(
            "run", "--no-capture-output", "-n", [string]$config.conda_env,
            "python", "-u", [string]$model.extractor.script
        )
        if ($model.extractor.mode -eq "argparse") {
            $extractArguments += @(
                "--image-dir", $GalleryRoot,
                "--model-path", $checkpoint,
                "--output-feats", $rawCache
            )
            $extractArguments += @($model.extractor.arguments | ForEach-Object { [string]$_ })
        }
        elseif ($model.extractor.mode -eq "override") {
            $extractArguments += @(
                "--override", "image_dir=$GalleryRoot",
                "--override", "model_path=$checkpoint",
                "--override", "output_feats=$rawCache"
            )
            foreach ($override in $model.extractor.arguments) {
                $extractArguments += @("--override", [string]$override)
            }
        }
        else { throw "Unsupported extractor mode for ${id}: $($model.extractor.mode)" }
        Invoke-CheckedCommand $workdir $extractArguments $logPath "${id}:extract"
    }

    if ($doStamp) {
        if (-not $DryRun -and -not (Test-Path -LiteralPath $rawCache -PathType Leaf)) {
            throw "Missing raw cache for ${id}: $rawCache"
        }
        $preprocessingJson = $model.preprocessing | ConvertTo-Json -Compress -Depth 10
        $ttaJson = $model.tta | ConvertTo-Json -Compress -Depth 10
        $environmentJson = @{ conda_env = [string]$config.conda_env } | ConvertTo-Json -Compress
        $preprocessingJson = ConvertTo-NativeJsonArgument $preprocessingJson
        $ttaJson = ConvertTo-NativeJsonArgument $ttaJson
        $environmentJson = ConvertTo-NativeJsonArgument $environmentJson
        $stampArguments = @(
            "run", "--no-capture-output", "-n", [string]$config.conda_env,
            "python", "-u",
            (Join-Path $RepoRoot "scripts\stamp_feature_cache.py"),
            "--env", $resolvedEnvFile,
            "--raw-cache", $rawCache,
            "--output", $validatedCache,
            "--model-id", $id,
            "--model-source", [string]$model.model_source,
            "--feature-normalization", "l2",
            "--checkpoint", $checkpoint,
            "--preprocessing-json", $preprocessingJson,
            "--tta-json", $ttaJson,
            "--environment-json", $environmentJson,
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
            "run", "--no-capture-output", "-n", [string]$config.conda_env,
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
