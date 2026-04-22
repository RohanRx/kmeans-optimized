import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Calculate GPU Speedup vs CPU Baseline")
    parser.add_argument("-d", "--dimensions", type=int, choices=[16, 512], default=16, help="Dimensions: 16 (K=256) or 512 (K=5)")
    parser.add_argument("-e", "--extra", type=str, default="", help="Suffix used in GPU filename (e.g., _test)")
    args = parser.parse_args()

    D = args.dimensions
    EXTRA = args.extra

    OUTPUT_DIR = os.path.join("output", f"D{D}")
    CPU_FILE = os.path.join(OUTPUT_DIR, "cpu_baseline.csv")
    GPU_FILE = os.path.join(OUTPUT_DIR, f"gpu_optimized{EXTRA}.csv")
    
    PLOT_OUT = os.path.join(OUTPUT_DIR, f"speedup_analysis{EXTRA}.png")
    TABLE_OUT = os.path.join(OUTPUT_DIR, f"speedup_results{EXTRA}.csv")

    if not os.path.exists(CPU_FILE) or not os.path.exists(GPU_FILE):
        print(f"Error: Missing input CSV files in {OUTPUT_DIR}")
        print(f"Looked for:\n  {CPU_FILE}\n  {GPU_FILE}")
        return

    df_cpu = pd.read_csv(CPU_FILE)
    df_gpu = pd.read_csv(GPU_FILE)

    df = pd.merge(
        df_cpu, 
        df_gpu, 
        on=['category', 'n', 'd', 'k', 'iters'], 
        suffixes=('_cpu', '_gpu')
    )
    df['speedup'] = df['ms_per_iter_cpu'] / df['ms_per_iter_gpu']

    plt.figure(figsize=(10, 6))
    
    for cat in df['category'].unique():
        subset = df[df['category'] == cat]
        plt.plot(subset['n'], subset['speedup'], marker='o', linewidth=2, label=cat.capitalize())

    plt.xscale('log')
    plt.yscale('linear')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.xlabel('Number of Points (N)')
    plt.ylabel('Speedup Factor (x)')
    plt.title(f'K-Means Performance: GPU Speedup over CPU (D={D})')
    
    # Baseline reference at 1x
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Break-even (1x)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_OUT)
    print(f"[✓] Speedup graph saved to: {PLOT_OUT}")

    report = df[['category', 'n', 'iters', 'ms_per_iter_cpu', 'ms_per_iter_gpu', 'speedup']].copy()
    report_formatted = report.copy()
    report_formatted['ms_per_iter_cpu'] = report_formatted['ms_per_iter_cpu'].map('{:.3f}'.format)
    report_formatted['ms_per_iter_gpu'] = report_formatted['ms_per_iter_gpu'].map('{:.3f}'.format)
    report_formatted['speedup'] = report_formatted['speedup'].map('{:.2f}x'.format)

    report_formatted.to_csv(TABLE_OUT, index=False)
    print(f"[✓] Comparison table saved to: {TABLE_OUT}")
    
    print("\n--- Speedup Metrics Summary ---")
    print(report_formatted.to_string(index=False))

    # Summary Stats
    avg_speedup = df['speedup'].mean()
    peak_speedup = df['speedup'].max()
    min_speedup = df['speedup'].min()

    print("\n--- GPU Acceleration Summary ---")
    print(f"Max Speedup:     {peak_speedup:.2f}x")
    print(f"Minimum Speedup: {min_speedup:.2f}x")
    print(f"Average Speedup: {avg_speedup:.2f}x")

if __name__ == "__main__":
    main()