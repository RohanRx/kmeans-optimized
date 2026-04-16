import numpy as np
from sklearn.datasets import make_blobs
import os

OUTPUT_DIR = "datasets"

# Hardcoded parameters for the single dataset
N = 240 * (2 ** 10)  # 245,760 samples
D = 16               # 16 dimensions
K = 256              # 256 clusters

os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_data(n_samples, n_features, n_clusters):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=42
    )
    return X.astype(np.float32)

print(f"Generating single dataset: N={N}, D={D}, K={K}")

# Generate the dataset
X = create_data(N, D, K)
filename = f"{OUTPUT_DIR}/blobs_N{N}_D{D}_K{K}.bin"

# Save as raw binary file
X.tofile(filename) 

print(f"Successfully saved to {filename}")
print("\nComplete")