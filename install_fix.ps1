# Strategy: Redefine the 'rsproxy' source that is used in the user's config.toml
# Instead of pointing to the git repo (which 404s), point it to the sparse index.
$env:CARGO_SOURCE_RSPROXY_REGISTRY = 'sparse+https://rsproxy.cn/index/'

# Also try to force crates-io to use sparse directly, overriding the replacement if possible
# (though replace-with usually wins)
$env:CARGO_REGISTRIES_CRATES_IO_PROTOCOL = 'sparse'

# Remove Conda env vars to satisfy maturin
Remove-Item Env:CONDA_PREFIX -ErrorAction SilentlyContinue
Remove-Item Env:CONDA_DEFAULT_ENV -ErrorAction SilentlyContinue

# Clean cargo cache just in case (optional, but good)
# Remove-Item -Recurse -Force "$env:USERPROFILE\.cargo\registry\index" -ErrorAction SilentlyContinue

Write-Host "Installing pyfgsea with overridden registry..."
pip install -v .
