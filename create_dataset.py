import numpy as np
from sklearn.datasets import make_blobs
import os

DIMENSIONS = [16, 512]
BASE = 240
START_POWER = 4
OUTPUT_DIR = "datasets"

# dimension-specific K
K_VALUES = {
    16: 256,
    512: 24
}

# dimension-specific limits
MAX_POINTS = {
    16: 16_000_000,   # allow up to ~15.7M (>10M)
    512: 1_000_000    # stop near 1M
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# generate sizes
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

def create_data(n_samples, n_features, n_clusters):
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        random_state=42
    )
    return X.astype(np.float32)

for D in DIMENSIONS:
    sizes = get_sizes(D)
    K = K_VALUES[D]   # pick correct K per dimension

    print(f"\nDimension {D} (K={K}) sizes:")
    print(sizes)

    for N in sizes:
        print(f"Generating N={N}, D={D}, K={K}")

        X = create_data(N, D, K)
        filename = f"{OUTPUT_DIR}/blobs_N{N}_D{D}_K{K}.bin"

        X.tofile(filename)  # save as raw binary file

print("\nComplete")