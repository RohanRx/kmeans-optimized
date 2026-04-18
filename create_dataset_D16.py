import numpy as np
from sklearn.datasets import make_blobs
import os

D = 16
K = 256

BASE_DIR = f"data_D{D}"
EASY_DIR = os.path.join(BASE_DIR, "easy")
HARD_DIR = os.path.join(BASE_DIR, "hard")

BASE = 240
START_N = BASE * (2 ** 4)  # 3840

MAX_BYTES = 2 * (1024 ** 3)  # Stop once we reach a file > 2GB
OUTLIER_RATIO = 0.01
SEED = 42

np.random.seed(SEED)

os.makedirs(EASY_DIR, exist_ok=True)
os.makedirs(HARD_DIR, exist_ok=True)

def estimate_bytes(n_samples):
    return n_samples * D * 4

def create_easy(n_samples):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=D,
        centers=K,
        cluster_std=0.5,
        random_state=SEED
    )
    return X.astype(np.float32)

def create_hard(n_samples):
    num_outliers = int(OUTLIER_RATIO * n_samples)
    blob_samples = n_samples - num_outliers

    # create atomic contention on large clusters
    weights = np.logspace(0, 2, num=K) 
    weights /= weights.sum()
    sizes = np.random.multinomial(blob_samples, weights)

    X_list = []
    for i, size in enumerate(sizes):
        std_val = np.random.uniform(2.0, 8.0) 
        
        # variance with random_state
        Xi, _ = make_blobs(
            n_samples=size,
            n_features=D,
            centers=1,
            cluster_std=std_val,
            random_state=i 
        )
        X_list.append(Xi)

    X = np.vstack(X_list)

    # Add global outliers
    outliers = np.random.uniform(X.min(), X.max(), size=(num_outliers, D))
    X = np.vstack([X, outliers])

    return X.astype(np.float32)

def main():
    n = START_N

    print(f"Dataset Generation Started (Limit: {MAX_BYTES / (1024**3):.1f} GB)")
    print("-" * 50)

    while True:
        size_bytes = estimate_bytes(n)

        if size_bytes > MAX_BYTES:
            print(f"\nFinal target size reached. Stopping at {n} samples.")
            break

        print(f"\nProcessing N={n:,} (~{size_bytes / (1024**2):.2f} MB)")

        X_easy = create_easy(n)
        easy_filename = f"blobs_N{n}_D{D}_K{K}.bin"
        easy_path = os.path.join(EASY_DIR, easy_filename)
        X_easy.tofile(easy_path)
        print(f"  [✓] Written: {easy_path}")

        X_hard = create_hard(n)
        hard_filename = f"blobs_N{n}_D{D}_K{K}.bin"
        hard_path = os.path.join(HARD_DIR, hard_filename)
        X_hard.tofile(hard_path)
        print(f"  [✓] Written: {hard_path}")

        n *= 2

    print("-" * 50)
    print("Generation Complete. Files are located in 'data/easy' and 'data/hard'.")

if __name__ == "__main__":
    main()