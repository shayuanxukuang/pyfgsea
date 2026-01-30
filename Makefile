.PHONY: check lint test build

check: lint test

lint:
	ruff check pyfgsea repro
	ruff format pyfgsea repro
	mypy pyfgsea --ignore-missing-imports

test:
	pytest -q

build:
	maturin build --release
