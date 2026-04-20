import pandas as pd
import matplotlib.pyplot as plt
import os

CPU_FILE = "output/cpu_baseline.csv"
GPU_FILE = "output/gpu_optimized.csv"
PLOT_OUT = "output/speedup_analysis.png"
TABLE_OUT = "output/speedup_results.csv"

def main():
    if not os.path.exists(CPU_FILE) or not os.path.exists(GPU_FILE):
        print("Error: Missing input CSV files in output directory.")
        return

    # Load data
    df_cpu = pd.read_csv(CPU_FILE)
    df_gpu = pd.read_csv(GPU_FILE)

    # Merge dataframes on the problem dimensions
    df = pd.merge(
        df_cpu, 
        df_gpu, 
        on=['category', 'n', 'd', 'k', 'iters'], 
        suffixes=('_cpu', '_gpu')
    )

    # Calculate Speedup: (CPU Latency) / (GPU Latency)
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
    plt.title('K-Means Performance: GPU Speedup over CPU Baseline')
    
    # Add a baseline reference line at 1x
    plt.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Break-even (1x)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_OUT)
    print("[✓] Speedup graph saved to: {}".format(PLOT_OUT))

    # Select and rename columns for a clean final report
    report = df[['category', 'n', 'iters', 'ms_per_iter_cpu', 'ms_per_iter_gpu', 'speedup']].copy()
    
    # Format numbers for readability
    report['ms_per_iter_cpu'] = report['ms_per_iter_cpu'].map('{:.3f}'.format)
    report['ms_per_iter_gpu'] = report['ms_per_iter_gpu'].map('{:.3f}'.format)
    report['speedup'] = report['speedup'].map('{:.2f}x'.format)

    report.to_csv(TABLE_OUT, index=False)
    print("[✓] Comparison table saved to: {}".format(TABLE_OUT))
    
    # Print the table to terminal
    print("\n--- Speedup Metrics Summary ---")
    print(report.to_string(index=False))

    # Calculate different types of speedup metrics
    avg_speedup = df['speedup'].mean()
    peak_speedup = df['speedup'].max()
    min_speedup = df['speedup'].min()

    print("\n--- GPU Acceleration Summary ---")
    print("Max Speedup:    {:.2f}x".format(peak_speedup))
    print("Minimum Speedup: {:.2f}x".format(min_speedup))
    print("Average Speedup: {:.2f}x".format(avg_speedup))

if __name__ == "__main__":
    main()