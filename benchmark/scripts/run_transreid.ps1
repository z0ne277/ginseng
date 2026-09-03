[CmdletBinding()]
param(
    [string]$EnvFile = ".env",
    [ValidateSet("all", "extract", "stamp", "evaluate")]
    [string]$Phase = "all",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ProtocolPath = Join-Path $RepoRoot "artifacts\manifests\query_groups.json"
$RawCache = Join-Path $RepoRoot "artifacts\features\raw\transreid_external_checkpoint_271_1075.pt"
$MetadataPath = Join-Path $RepoRoot "artifacts\features\raw\transreid_external_checkpoint_271_1075.metadata.json"
$ValidatedCache = Join-Path $RepoRoot "artifacts\features\validated\transreid_external_checkpoint_271_1075.npz"
$ResultJson = Join-Path $RepoRoot "artifacts\results\transreid_external_checkpoint_271_1075.json"
$ResultCsv = Join-Path $RepoRoot "artifacts\results\transreid_external_checkpoint_271_1075_per_query.csv"
$LogPath = Join-Path $RepoRoot "artifacts\logs\transreid_external_checkpoint_271_1075.log"
$CondaEnv = "ginseng-baselines"

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

function Require-Path {
    param([hashtable]$Values, [string]$Key, [string]$PathType)
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

function Invoke-Checked {
    param([string[]]$Arguments, [string]$Label)
    Write-Host "[$Label] $(Format-Command $Arguments)"
    Write-Host "[$Label] log=$LogPath"
    if ($DryRun) { return }
    New-Item -ItemType Directory -Path (Split-Path -Parent $LogPath) -Force | Out-Null
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exitCode = 1
    try {
        & conda @Arguments 2>&1 | Tee-Object -FilePath $LogPath -Append
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($exitCode -ne 0) { throw "$Label failed with exit code $exitCode" }
}

function ConvertTo-NativeJsonArgument {
    param([string]$Json)
    return $Json -replace '"', '\"'
}

$resolvedEnv = if ([IO.Path]::IsPathRooted($EnvFile)) { $EnvFile } else { Join-Path $RepoRoot $EnvFile }
$values = Read-DotEnv $resolvedEnv
$GalleryRoot = Require-Path $values "MERGED_GALLERY" "directory"
$TransReIDRoot = Require-Path $values "TRANSREID_ROOT" "directory"
$ConfigFile = Require-Path $values "TRANSREID_CONFIG" "file"
$Checkpoint = Require-Path $values "TRANSREID_CHECKPOINT" "file"
if (-not $DryRun -and -not (Test-Path -LiteralPath $ProtocolPath -PathType Leaf)) {
    throw "Missing canonical query protocol: $ProtocolPath"
}

if ($Phase -in @("all", "extract")) {
    $arguments = @(
        "run", "--no-capture-output", "-n", $CondaEnv,
        "python", "-u", (Join-Path $RepoRoot "scripts\extract_transreid.py"),
        "--transreid-root", $TransReIDRoot,
        "--config-file", $ConfigFile,
        "--checkpoint", $Checkpoint,
        "--image-dir", $GalleryRoot,
        "--output", $RawCache,
        "--metadata-output", $MetadataPath,
        "--expected-count", "12787",
        "--batch-size", "64",
        "--num-workers", "4",
        "--device", "auto",
        "--trusted-local-checkpoint"
    )
    Invoke-Checked $arguments "transreid:extract"
}

if ($Phase -in @("all", "stamp")) {
    if (-not $DryRun -and (-not (Test-Path $RawCache) -or -not (Test-Path $MetadataPath))) {
        throw "Missing TransReID raw cache or metadata sidecar"
    }
    if ($DryRun) {
        $preprocessingJson = '{"source":"metadata sidecar from official config"}'
        $ttaJson = '{"enabled":false,"weights":[1.0]}'
        $environmentJson = '{"conda_env":"ginseng-baselines","official_commit":"dec55046fcdfadee14e2c28e2df89305d8f7557a"}'
        $featureDim = $null
    }
    else {
        $metadata = Get-Content -LiteralPath $MetadataPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $preprocessingJson = $metadata.preprocessing | ConvertTo-Json -Compress -Depth 10
        $ttaJson = $metadata.tta | ConvertTo-Json -Compress -Depth 10
        $environmentJson = @{
            conda_env = $CondaEnv
            official_commit = [string]$metadata.official_commit
            config_name = [string]$metadata.config_name
            jpm = [bool]$metadata.jpm
            sie_camera = [bool]$metadata.sie_camera
            sie_view = [bool]$metadata.sie_view
        } | ConvertTo-Json -Compress
        $featureDim = [string]$metadata.feature_dim
    }
    $preprocessingJson = ConvertTo-NativeJsonArgument $preprocessingJson
    $ttaJson = ConvertTo-NativeJsonArgument $ttaJson
    $environmentJson = ConvertTo-NativeJsonArgument $environmentJson
    $arguments = @(
        "run", "--no-capture-output", "-n", $CondaEnv,
        "python", "-u", (Join-Path $RepoRoot "scripts\stamp_feature_cache.py"),
        "--env", $resolvedEnv,
        "--raw-cache", $RawCache,
        "--output", $ValidatedCache,
        "--model-id", "transreid_external_checkpoint",
        "--model-source", "official-transreid@dec55046fcdfadee14e2c28e2df89305d8f7557a",
        "--feature-normalization", "l2",
        "--checkpoint", $Checkpoint,
        "--preprocessing-json", $preprocessingJson,
        "--tta-json", $ttaJson,
        "--environment-json", $environmentJson,
        "--trusted-local-pt"
    )
    if ($featureDim) { $arguments += @("--expected-feature-dim", $featureDim) }
    Invoke-Checked $arguments "transreid:stamp"
}

if ($Phase -in @("all", "evaluate")) {
    if (-not $DryRun -and -not (Test-Path $ValidatedCache)) {
        throw "Missing validated TransReID cache: $ValidatedCache"
    }
    $arguments = @(
        "run", "--no-capture-output", "-n", $CondaEnv,
        "python", "-u", (Join-Path $RepoRoot "scripts\evaluate_features.py"),
        "--cache", $ValidatedCache,
        "--query-groups", $ProtocolPath,
        "--output", $ResultJson,
        "--per-query-csv", $ResultCsv,
        "--ks", "1,5,10,20",
        "--block-size", "32",
        "--bootstrap-iterations", "2000",
        "--bootstrap-seed", "42"
    )
    Invoke-Checked $arguments "transreid:evaluate"
}

Write-Host "Completed TransReID phase=$Phase dry_run=$([bool]$DryRun)"
