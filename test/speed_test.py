import time
import numpy as np
from sklearn.cluster import KMeans

REPEATS = 1
ITERATIONS = 1000

def load_bin(filename, n_samples, n_features):
    X = np.fromfile(filename, dtype=np.float32)
    return X.reshape(n_samples, n_features)

def run_benchmark(filename, n_samples, n_features, K):
    X = load_bin(filename, n_samples, n_features)

    initial_centroids = X[:K] 

    times = []

    for _ in range(REPEATS):
        model = KMeans(
            n_clusters=K,
            init=initial_centroids,
            n_init=1,
            max_iter=ITERATIONS,
            algorithm='lloyd',
            random_state=42
        )

        # model = KMeans(
        #     n_clusters=K,
        #     init=initial_centroids,
        #     n_init=1,
        #     max_iter=ITERATIONS,
        #     algorithm='lloyd',
        #     random_state=42,
        #     tol=1e-6 # Set to a small value to see actual convergence
        # )

        start = time.time()
        model.fit(X)
        end = time.time()

        times.append(end - start)

    avg = np.mean(times)
    
    print(f"N: {n_samples} | D: {n_features} | K: {K}")
    print(f"Avg Time: {avg:.4f}s")
    print(f"Converged in: {model.n_iter_} iterations") 

    counts = np.bincount(model.labels_, minlength=K)

    print("\n--- Centroid Stats ---")
    for i in range(K):
        print(f"Centroid {i}: {counts[i]} points")
        coords = ", ".join(f"{v:.4f}" for v in model.cluster_centers_[i])
        print(f"Coordinates: [ {coords} ]\n")

    return avg

def main():
    run_benchmark("sample_datasets/blobs_N245760_D16_K256.bin", 245760, 16, 256)

if __name__ == "__main__":
    main()