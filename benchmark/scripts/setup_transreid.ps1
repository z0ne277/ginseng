[CmdletBinding()]
param(
    [string]$Destination = "third_party\TransReID",
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Repository = "https://github.com/damo-cv/TransReID.git"
$PinnedCommit = "dec55046fcdfadee14e2c28e2df89305d8f7557a"
$Target = if ([IO.Path]::IsPathRooted($Destination)) {
    $Destination
} else {
    Join-Path $RepoRoot $Destination
}

Write-Host "TransReID source=$Repository"
Write-Host "Pinned commit=$PinnedCommit"
Write-Host "Destination=$Target"
if ($DryRun) {
    Write-Host "git clone $Repository $Target"
    Write-Host "git -C $Target checkout --detach $PinnedCommit"
    return
}

if (-not (Test-Path -LiteralPath $Target)) {
    & git clone $Repository $Target
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
}
elseif (-not (Test-Path -LiteralPath (Join-Path $Target ".git"))) {
    throw "Destination exists but is not a Git repository: $Target"
}

& git -C $Target fetch origin $PinnedCommit --depth 1
if ($LASTEXITCODE -ne 0) { throw "git fetch failed" }
& git -C $Target checkout --detach $PinnedCommit
if ($LASTEXITCODE -ne 0) { throw "git checkout failed" }
$Head = (& git -C $Target rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $Head -ne $PinnedCommit) {
    throw "TransReID HEAD mismatch: expected $PinnedCommit, found $Head"
}
Write-Host "TransReID setup passed: HEAD=$Head"
