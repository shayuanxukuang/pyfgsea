param(
    [Parameter(Mandatory = $true)]
    [string]$ReleaseTag,

    [Parameter(Mandatory = $true)]
    [string]$EvidenceRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$evidencePath = [System.IO.Path]::GetFullPath($EvidenceRoot)
$relativeEvidencePath = [System.IO.Path]::GetRelativePath($repoRoot, $evidencePath)
$parentPrefix = '..' + [System.IO.Path]::DirectorySeparatorChar

if (
    $relativeEvidencePath -eq '.' -or
    -not $relativeEvidencePath.StartsWith($parentPrefix, [System.StringComparison]::Ordinal)
) {
    throw 'EvidenceRoot must be outside the verified Git worktree.'
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

$distPath = Join-Path $evidencePath 'dist'
$venvPath = Join-Path $evidencePath 'venv'
$workPath = Join-Path $evidencePath 'work'
$evidenceDirectory = Join-Path $evidencePath 'evidence'
$receiptPath = Join-Path $evidenceDirectory 'receipt.json'

python (Join-Path $repoRoot 'scripts\verify_pyfgsea_artifacts.py') `
    --repo $repoRoot `
    --commit $commit `
    --release-tag $ReleaseTag `
    --output-dir $distPath `
    --venv $venvPath `
    --work-dir $workPath `
    --receipt $receiptPath
if ($LASTEXITCODE -ne 0) { throw 'Local sdist-to-wheel verification failed.' }

Write-Host 'Local artifact-chain and installed-test verification passed.' -ForegroundColor Green
Write-Host "Receipt: $receiptPath"
Write-Host 'This is not the cross-platform, reference-OCI, Figure 1, Figure 2, or manuscript-closure gate.' -ForegroundColor Yellow
