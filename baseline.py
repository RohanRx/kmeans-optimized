import time
import numpy as np
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt

DIMENSIONS = [16, 512]

BASE = 240
START_POWER = 4
ITERATIONS = 100
REPEATS = 1
DATA_DIR = "datasets"

# dimension-specific K
K_VALUES = {
    16: 256,
    512: 24
}

# dimension-specific limits
MAX_POINTS = {
    16: 16_000_000,
    512: 1_000_000
}

def get_sizes(D):
    sizes = []
    i = START_POWER

    while True:
        n = BASE * (2 ** i)
        if n > MAX_POINTS[D]:
            break
        sizes.append(n)
        i += 1

    return sizes

def load_bin(filename, n_samples, n_features):
    X = np.fromfile(filename, dtype=np.float32)
    return X.reshape(n_samples, n_features)

def run_benchmark(filename, n_samples, n_features, K):
    X = load_bin(filename, n_samples, n_features)

    times = []

    for _ in range(REPEATS):
        model = KMeans(
            n_clusters=K,
            init='random',
            n_init=1,
            max_iter=ITERATIONS,
            algorithm='lloyd',
            random_state=42
        )

        start = time.time()
        model.fit(X)
        end = time.time()

        times.append(end - start)

    avg = np.mean(times)
    std = np.std(times)

    print(f"N: {n_samples:>9} | D: {n_features:>3} | K: {K:>3} | "f"Avg: {avg:.4f}s | Std: {std:.4f}s")

    return avg

def main():
    for D in DIMENSIONS:
        K = K_VALUES[D]
        sizes = get_sizes(D)
        results = []

        print(f"\n===== Dimension {D} (K={K}) =====")

        for N in sizes:
            filename = os.path.join(
                DATA_DIR,
                f"blobs_N{N}_D{D}_K{K}.bin"
            )

            avg = run_benchmark(filename, N, D, K)
            results.append(avg)

        plt.figure()

        plt.plot(sizes, results, marker='o')
        plt.xlabel("Number of Points (N)")
        plt.ylabel("Time (seconds)")
        plt.title(f"CPU KMeans Baseline (D={D}, K={K})")
        plt.grid()

        output_name = f"baseline_cpu_D{D}.png"
        plt.savefig(output_name)
        plt.show()

if __name__ == "__main__":
    main()