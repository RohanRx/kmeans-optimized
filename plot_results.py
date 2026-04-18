import pandas as pd
import matplotlib.pyplot as plt
import os

CSV_INPUT = "output/cpu_baseline.csv"
PLOT_OUTPUT = "output/baseline_D16.png"

def main():
    if not os.path.exists(CSV_INPUT):
        print("Error: CSV file not found at {}".format(CSV_INPUT))
        return

    df = pd.read_csv(CSV_INPUT)

    plt.figure(figsize=(10, 6))

    for cat in df['category'].unique():
        subset = df[df['category'] == cat]
        plt.plot(subset['n'], subset['ms_per_iter'], marker='o', label=cat.capitalize())

    plt.xlabel("Number of Points (N)")
    plt.ylabel("Latency (ms / iteration)")
    plt.title("K-Means CPU Baseline: Latency vs Points")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()

    plt.savefig(PLOT_OUTPUT)
    print("[✓] Plot saved to: {}".format(PLOT_OUTPUT))
    plt.show()

if __name__ == "__main__":
    main()