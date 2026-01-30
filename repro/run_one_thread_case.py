import os
import sys
import time
import psutil
import threading
from pathlib import Path

# Set threads BEFORE importing libraries that might init rayon/numpy
if len(sys.argv) > 1:
    os.environ["RAYON_NUM_THREADS"] = sys.argv[1]

# Ensure we can import pyfgsea
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

import pyfgsea  # noqa: E402
from repro.data_utils import generate_test_data  # noqa: E402


def monitor_memory(pid, stop_event, max_mem, base_mem):
    process = psutil.Process(pid)
    while not stop_event.is_set():
        try:
            mem = process.memory_info().rss
            for child in process.children(recursive=True):
                mem += child.memory_info().rss
            max_mem[0] = max(max_mem[0], mem - base_mem)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.01)


def main():
    if len(sys.argv) < 4:
        print("Usage: python run_one_thread_case.py <n_threads> <n_genes> <n_sets>")
        sys.exit(1)

    # n_threads = int(sys.argv[1])
    n_genes = int(sys.argv[2])
    n_sets = int(sys.argv[3])

    df_rnk, gmt_dict = generate_test_data(n_genes, n_sets, seed=42)

    # Setup memory monitoring
    process = psutil.Process(os.getpid())
    base_mem = process.memory_info().rss
    stop_event = threading.Event()
    peak_inc = [0]

    t = threading.Thread(
        target=monitor_memory,
        args=(os.getpid(), stop_event, peak_inc, base_mem),
        daemon=True,
    )
    t.start()

    # Run
    start_time = time.time()
    pyfgsea.run_gsea(df_rnk, gmt_dict, gene_col="Gene", score_col="Score", seed=42)
    elapsed = time.time() - start_time

    stop_event.set()
    t.join()

    peak_mb = peak_inc[0] / 1024 / 1024
    print(f"Time: {elapsed:.4f}, Memory: {peak_mb:.4f}")


if __name__ == "__main__":
    main()
