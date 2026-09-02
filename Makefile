.PHONY: check build

check:
	cargo fmt --check
	cargo test --locked
	python -m ruff check pyfgsea tests
	python -m ruff format --check pyfgsea tests
	python -m mypy pyfgsea --ignore-missing-imports
	python -m pytest -q

build:
	maturin build --release --locked
