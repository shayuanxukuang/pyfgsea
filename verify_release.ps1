Write-Host "Starting release verification..." -ForegroundColor Cyan

# 1. Static Checks
Write-Host "`n[1/5] Running static checks..."
ruff check pyfgsea repro
if ($LASTEXITCODE -ne 0) { Write-Error "Ruff check failed"; exit 1 }

mypy pyfgsea --ignore-missing-imports
if ($LASTEXITCODE -ne 0) { Write-Error "Mypy check failed"; exit 1 }

# 2. Unit Tests
Write-Host "`n[2/5] Running unit tests..."
# We run from parent dir to avoid import conflicts if we haven't installed editable, 
# but here we rely on the installed package or local source. 
# Best practice: run against INSTALLED package if possible, but for dev checks source is okay.
pytest -q tests
if ($LASTEXITCODE -ne 0) { Write-Error "Pytest failed"; exit 1 }

# 3. Build Wheel
Write-Host "`n[3/5] Building wheel..."
maturin build --release
if ($LASTEXITCODE -ne 0) { Write-Error "Maturin build failed"; exit 1 }

# 4. Install Wheel (Force Reinstall)
Write-Host "`n[4/5] Installing wheel..."
$wheels = Get-ChildItem "target/wheels/*.whl" | Sort-Object LastWriteTime -Descending
if (!$wheels) { Write-Error "No wheels found"; exit 1 }
$latest_wheel = $wheels[0].FullName
Write-Host "Installing: $latest_wheel"

pip install --force-reinstall $latest_wheel
if ($LASTEXITCODE -ne 0) { Write-Error "Pip install failed"; exit 1 }

# 5. Sanity Check (Outside Source Dir)
Write-Host "`n[5/5] Running sanity check in clean context..."
$temp_dir = [System.IO.Path]::GetTempPath()
Push-Location $temp_dir
try {
    python -c "import pyfgsea; import pandas as pd; print(f'Successfully imported pyfgsea v{pyfgsea.__version__}'); res = pyfgsea.run_gsea(pd.DataFrame({'g':['A','B'],'s':[1.,2.]}), {'P':['A']}, min_size=1, max_size=10); print('Toy run passed')"
    if ($LASTEXITCODE -ne 0) { Write-Error "Sanity check failed"; exit 1 }
} finally {
    Pop-Location
}

Write-Host "`n✅ VERIFICATION SUCCESSFUL! Ready for release." -ForegroundColor Green
