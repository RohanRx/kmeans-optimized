import numpy as np
from sklearn.datasets import make_blobs
import os
import argparse

def estimate_bytes(n_samples, d):
    return n_samples * d * 4

def create_easy(n_samples, d, k, seed):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=d,
        centers=k,
        cluster_std=0.5,
        random_state=seed
    )
    return X.astype(np.float32)

def create_hard(n_samples, d, k, outlier_ratio):
    num_outliers = int(outlier_ratio * n_samples)
    blob_samples = n_samples - num_outliers

    # create atomic contention on large clusters
    weights = np.logspace(0, 2, num=k) 
    weights /= weights.sum()
    sizes = np.random.multinomial(blob_samples, weights)

    X_list = []
    for i, size in enumerate(sizes):
        std_val = np.random.uniform(2.0, 8.0) 
        
        # variance with random_state
        Xi, _ = make_blobs(
            n_samples=size,
            n_features=d,
            centers=1,
            cluster_std=std_val,
            random_state=i 
        )
        X_list.append(Xi)

    X = np.vstack(X_list)

    # Add global outliers
    outliers = np.random.uniform(X.min(), X.max(), size=(num_outliers, d))
    X = np.vstack([X, outliers])

    return X.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Generate K-Means Datasets")
    parser.add_argument("-d", "--dimensions", type=int, choices=[16, 512], default=16, help="Dimensions: 16 (K=256) or 512 (K=5)")
    args = parser.parse_args()

    D = args.dimensions
    K = 256 if D == 16 else 5

    BASE_DATA_DIR = "data"
    DIM_DIR = os.path.join(BASE_DATA_DIR, f"D{D}")
    EASY_DIR = os.path.join(DIM_DIR, "easy")
    HARD_DIR = os.path.join(DIM_DIR, "hard")

    os.makedirs(EASY_DIR, exist_ok=True)
    os.makedirs(HARD_DIR, exist_ok=True)

    BASE = 240
    START_N = BASE * (2 ** 4)  # 3840
    MAX_BYTES = 2 * (1024 ** 3)  # Stop once we reach a file > 2GB
    OUTLIER_RATIO = 0.01
    SEED = 42

    np.random.seed(SEED)

    n = START_N

    print(f"Dataset Generation Started (Limit: {MAX_BYTES / (1024**3):.1f} GB)")
    print("-" * 50)

    while True:
        size_bytes = estimate_bytes(n, D)

        if size_bytes > MAX_BYTES:
            print(f"\nFinal target size reached. Stopping at {n} samples.")
            break

        print(f"\nProcessing N={n:,} (~{size_bytes / (1024**2):.2f} MB)")

        X_easy = create_easy(n, D, K, SEED)
        easy_filename = f"blobs_N{n}_D{D}_K{K}.bin"
        easy_path = os.path.join(EASY_DIR, easy_filename)
        X_easy.tofile(easy_path)
        print(f"  [✓] Written: {easy_path}")

        X_hard = create_hard(n, D, K, OUTLIER_RATIO)
        hard_filename = f"blobs_N{n}_D{D}_K{K}.bin"
        hard_path = os.path.join(HARD_DIR, hard_filename)
        X_hard.tofile(hard_path)
        print(f"  [✓] Written: {hard_path}")

        n *= 2

    print("-" * 50)
    print(f"Generation Complete. Files are in {DIM_DIR}")

if __name__ == "__main__":
    main()