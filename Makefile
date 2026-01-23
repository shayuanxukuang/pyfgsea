
.PHONY: install test clean figures benchmarks all

install:
	pip install -r requirements.txt
	maturin develop --release

test:
	python -c "import pyfgsea; print('PyFgsea imported successfully')"

benchmarks:
	@echo "Running performance benchmarks..."
	python repro/fig1_table1_performance.py
	python repro/benchmark_threads.py
	python repro/benchmark_calibration.py

figures:
	@echo "Generating figures..."
	python repro/fig_ablation_tail.py
	python repro/fig_stability.py

supp:
	@echo "Generating supplementary figures..."
	python repro/fig_supp_tail_consistency.py
	python repro/fig_supp_bland_altman.py
	python repro/fig_supp_thread_scaling.py
	python repro/fig_supp_null_calibration.py
	python repro/fig_supp_window_sensitivity.py
	python repro/fig_supp_myeloid_traj.py

all: benchmarks figures supp
	@echo "All reproduction scripts completed."

info:
	python scripts/print_versions.py
