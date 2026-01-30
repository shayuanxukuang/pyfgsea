Write-Host "Running Static Analysis..."
ruff check pyfgsea repro
ruff format pyfgsea repro
mypy pyfgsea --ignore-missing-imports

Write-Host "Running Tests..."
pytest -q

Write-Host "Building Release..."
maturin build --release
