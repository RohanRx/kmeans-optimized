import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot a single K-Means Benchmark CSV")
    parser.add_argument("-d", "--dimensions", type=int, choices=[16, 512], default=16, help="Dimensions folder to look in (16 or 512)")
    parser.add_argument("-f", "--filename", type=str, default="gpu_optimized.csv", help="The CSV filename to plot (e.g., cpu_baseline.csv or gpu_optimized.csv)")
    args = parser.parse_args()

    D = args.dimensions
    filename = args.filename

    INPUT_PATH = os.path.join("output", f"D{D}", filename)
    PLOT_NAME = f"plot_{os.path.splitext(filename)[0]}.png"
    PLOT_OUTPUT = os.path.join("output", f"D{D}", PLOT_NAME)

    if not os.path.exists(INPUT_PATH):
        print(f"Error: CSV file not found at {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)

    plt.figure(figsize=(10, 6))

    for cat in df['category'].unique():
        subset = df[df['category'] == cat]
        plt.plot(subset['n'], subset['ms_per_iter'], marker='o', label=cat.capitalize())

    plt.xlabel("Number of Points (N)")
    plt.ylabel("Latency (ms / iteration)")
    plt.title(f"K-Means Performance: {filename} (D={D})")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    plt.savefig(PLOT_OUTPUT)
    print(f"[✓] Plot saved to: {PLOT_OUTPUT}")
    plt.show()

if __name__ == "__main__":
    main()