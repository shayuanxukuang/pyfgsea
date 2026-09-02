.PHONY: check build

check:
	cargo fmt --check
	cargo test --locked
	python -m ruff check pyfgsea tests examples
	python -m ruff format --check pyfgsea tests examples
	python -m mypy pyfgsea --ignore-missing-imports
	python -m maturin develop --release --locked
	python -m pytest -q

build:
	maturin build --release --locked
