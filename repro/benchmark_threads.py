import sys
import subprocess
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def run_case(n_threads, n_genes, n_sets, script_path):
    cmd = [
        sys.executable, 
        str(script_path),
        str(n_threads),
        str(n_genes),
        str(n_sets)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error (threads={n_threads}): {result.stderr}")
        return None, None
    
    match = re.search(r"Time: ([\d\.]+), Memory: ([\d\.]+)", result.stdout)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None, None

def main():
    out_dir = Path("results/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    script_path = Path(__file__).parent / "run_one_thread_case.py"
    if not script_path.exists():
        print(f"Error: {script_path} not found.")
        return

    thread_counts = [1, 2, 4, 8, 16]
    n_genes = 20000
    n_sets = 5000
    
    results = []
    print(f"Benchmarking Thread Scaling (Genes={n_genes}, Sets={n_sets})...")
    print(f"{'Threads':<10} {'Time(s)':<10} {'PeakRSS(MB)':<10}")
    
    for t in thread_counts:
        time_s, mem_mb = run_case(t, n_genes, n_sets, script_path)
        if time_s is not None:
            print(f"{t:<10} {time_s:<10.4f} {mem_mb:<10.4f}")
            results.append({"Threads": t, "Time": time_s, "Memory": mem_mb})
            
    df = pd.DataFrame(results)
    csv_path = out_dir / "benchmark_thread_scaling.csv"
    df.to_csv(csv_path, index=False)
    
    # Visualization
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Threads')
    ax1.set_ylabel('Time (s)', color=color)
    ax1.plot(df['Threads'], df['Time'], marker='o', color=color, label='Time')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(thread_counts)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Peak Memory (MB)', color=color)
    ax2.plot(df['Threads'], df['Memory'], marker='s', linestyle='--', color=color, label='Memory')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f"Thread Scaling Performance\n(Genes={n_genes}, Sets={n_sets})")
    plt.tight_layout()
    
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)
    plt.savefig(fig_dir / "Figure_Thread_Scaling.png", dpi=300)
    print(f"Plot saved to {fig_dir / 'Figure_Thread_Scaling.png'}")

if __name__ == "__main__":
    main()
