param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseTag,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputRoot)
$relativeOutputPath = [System.IO.Path]::GetRelativePath($repoRoot, $outputPath)
$parentPrefix = '..' + [System.IO.Path]::DirectorySeparatorChar

if (
    $relativeOutputPath -eq '.' -or
    -not $relativeOutputPath.StartsWith($parentPrefix, [System.StringComparison]::Ordinal)
) {
    throw 'OutputRoot must be outside the verified Git worktree.'
}

$status = git -C $repoRoot status --porcelain=v1 --untracked-files=all
if ($LASTEXITCODE -ne 0) { throw 'Unable to inspect the Git worktree.' }
if ($status) { throw 'Release verification requires a clean Git worktree.' }

$commit = (git -C $repoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Unable to resolve HEAD.' }
$tagRef = "refs/tags/$ReleaseTag"
$tagType = (git -C $repoRoot cat-file -t $tagRef).Trim()
if ($LASTEXITCODE -ne 0 -or $tagType -ne 'tag') {
    throw "ReleaseTag must name an annotated tag: $ReleaseTag"
}
$tagCommit = (git -C $repoRoot rev-parse "$tagRef^{commit}").Trim()
if ($LASTEXITCODE -ne 0 -or $tagCommit -ne $commit) {
    throw "ReleaseTag does not peel to HEAD: $ReleaseTag"
}

$distPath = Join-Path $outputPath 'dist'
$venvPath = Join-Path $outputPath 'venv'
$workPath = Join-Path $outputPath 'work'
$reportDirectory = Join-Path $outputPath 'evidence'
$reportPath = Join-Path $reportDirectory 'receipt.json'

python (Join-Path $repoRoot 'scripts\verify_pyfgsea_artifacts.py') `
    --repo $repoRoot `
    --commit $commit `
    --release-tag $ReleaseTag `
    --output-dir $distPath `
    --venv $venvPath `
    --work-dir $workPath `
    --receipt $reportPath
if ($LASTEXITCODE -ne 0) { throw 'Local sdist-to-wheel verification failed.' }

Write-Host 'Local package checks passed.' -ForegroundColor Green
Write-Host "Report: $reportPath"
Write-Host 'Paper figures are checked separately.'
