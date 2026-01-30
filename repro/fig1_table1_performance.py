import os
import sys
import time
import psutil
import threading
import pandas as pd
from pathlib import Path

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from repro.data_utils import generate_test_data  # noqa: E402

# Optional imports
try:
    import pyfgsea
except ImportError:
    print("Warning: pyfgsea not found.")

try:
    import gseapy as gp
except ImportError:
    gp = None

try:
    import blitzgsea
except ImportError:
    blitzgsea = None


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
        time.sleep(0.05)


def run_tool(tool_name, df_rnk, gmt_obj):
    if tool_name == "PyFgsea":
        pyfgsea.run_gsea(df_rnk, gmt_obj, gene_col="Gene", score_col="Score", seed=42)
    elif tool_name == "GSEApy":
        if gp is None:
            return
        gp.prerank(
            rnk=df_rnk,
            gene_sets=gmt_obj,
            threads=4,
            min_size=15,
            max_size=2000,
            seed=42,
            verbose=False,
        )
    elif tool_name == "BlitzGSEA":
        if blitzgsea is None:
            return
        sig_copy = df_rnk[["Gene", "Score"]].copy()
        # BlitzGSEA uses multiprocessing
        blitzgsea.gsea(
            sig_copy,
            gmt_obj,
            min_size=15,
            max_size=2000,
            seed=42,
            verbose=False,
            processes=4,
        )


def main():
    # Enforce thread consistency
    for env_var in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]:
        os.environ[env_var] = "4"

    out_dir = Path("results/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = out_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    scenarios = [
        {"genes": 12000, "sets": 200, "desc": "Small"},
        {"genes": 20000, "sets": 1000, "desc": "Medium"},
        {"genes": 20000, "sets": 5000, "desc": "Large"},
    ]

    tools = ["PyFgsea", "GSEApy", "BlitzGSEA"]
    results = []

    print(f"{'Dataset':<10} {'Tool':<10} {'Time(s)':<10} {'RAM(MB)':<10}")

    for scen in scenarios:
        n_g, n_s = scen["genes"], scen["sets"]
        df_rnk, gmt_dict = generate_test_data(n_g, n_s, seed=42)

        # Write GMT for GSEApy
        gmt_file = temp_dir / f"temp_{n_g}_{n_s}.gmt"
        with open(gmt_file, "w") as f:
            for k, v in gmt_dict.items():
                joined_v = "\t".join(v)
                f.write(f"{k}\tNA\t{joined_v}\n")

        for tool in tools:
            # Force GC
            import gc

            gc.collect()

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

            # Warmup
            time.sleep(0.05)

            t0 = time.time()
            status = "Success"
            try:
                if tool == "GSEApy":
                    run_tool(tool, df_rnk, str(gmt_file))
                else:
                    run_tool(tool, df_rnk, gmt_dict)
            except Exception:
                # print(f"Error running {tool}: {e}")
                status = "Failed"

            t1 = time.time()

            stop_event.set()
            t.join()

            ram_mb = peak_inc[0] / 1024 / 1024
            duration = t1 - t0

            if status == "Success":
                print(
                    f"{scen['desc']:<10} {tool:<10} {duration:<10.2f} {ram_mb:<10.2f}"
                )
            else:
                print(f"{scen['desc']:<10} {tool:<10} {'FAILED':<10} {'-':<10}")

            results.append(
                {
                    "Dataset": scen["desc"],
                    "Genes": n_g,
                    "Sets": n_s,
                    "Tool": tool,
                    "Time_s": duration,
                    "RAM_MB": ram_mb,
                    "Status": status,
                }
            )

    pd.DataFrame(results).to_csv(out_dir / "performance_table.csv", index=False)


if __name__ == "__main__":
    main()
