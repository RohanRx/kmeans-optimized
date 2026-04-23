from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DIMENSIONS = 16

base_dir = Path(__file__).resolve().parent
data_dir = (base_dir / f"../output/D{DIMENSIONS}/block_test").resolve()
files = sorted(data_dir.glob("gpu_optimized_*.csv"))

if not files:
    raise FileNotFoundError(f"No matching CSV files found in {data_dir}")

out_dir = data_dir

categories = ["easy", "hard"]

for cat in categories:
    plt.figure(figsize=(10, 6))

    for file in files:
        df = pd.read_csv(file)

        label = file.stem.replace("gpu_optimized_", "")

        sub = df[df["category"] == cat]
        if sub.empty:
            continue

        sub = sub.sort_values("n")

        plt.plot(
            sub["n"],
            sub["ms_per_iter"],
            marker="o",
            linewidth=1.5,
            label=label
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of Points (N)")
    plt.ylabel("Latency (ms / iteration)")
    plt.title(f"GPU Threads Per Block Performance D{DIMENSIONS} ({cat.capitalize()} Dataset)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend(title="Values")

    output_path = out_dir / f"gpu_tpb_{cat}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    print(f"Saved plot to: {output_path}")

    plt.show()