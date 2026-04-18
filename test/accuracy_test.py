import time
import numpy as np

REPEATS = 1
ITERATIONS = 1000 

def load_bin(filename, n_samples, n_features):
    return np.fromfile(filename, dtype=np.float32).reshape(n_samples, n_features)

def pure_lloyd_kmeans(X, K, max_iter, initial_centroids):
    centroids = initial_centroids.copy()
    n_samples = X.shape[0]
    
    for i in range(max_iter):
        old_centroids = centroids.copy()
        
        # 1. Assignment Step
        # Using expanded square: dist(x,y)^2 = x^2 + y^2 - 2xy
        dist_sq = np.sum(X**2, axis=1)[:, np.newaxis] + \
                  np.sum(centroids**2, axis=1) - \
                  2 * np.dot(X, centroids.T)
        
        labels = np.argmin(dist_sq, axis=1)
        
        # 2. Update Step
        new_centroids = np.zeros_like(centroids)
        counts = np.bincount(labels, minlength=K)
        
        for k in range(K):
            if counts[k] > 0:
                new_centroids[k] = np.mean(X[labels == k], axis=0)
            else:
                # PURE LLOYD'S: No relocation. 
                # Centroid stays exactly where it was if no points are assigned.
                new_centroids[k] = centroids[k]
        
        centroids = new_centroids
        
        # Convergence Check
        if np.array_equal(centroids, old_centroids):
            print(f"   >> Converged early at iteration {i+1}")
            break
        
        if (i + 1) % 10 == 0:
            print(f"   Iteration {i+1}/{max_iter} completed...")

    return centroids, labels, counts

def run_benchmark(filename, n_samples, n_features, K):
    X = load_bin(filename, n_samples, n_features)
    
    initial_centroids = X[:K]

    times = []
    
    print(f"Starting Pure Lloyd's Benchmark (First-K Init, Max {ITERATIONS} iterations)...")
    
    for r in range(REPEATS):
        start = time.time()
        centroids, labels, counts = pure_lloyd_kmeans(X, K, ITERATIONS, initial_centroids)
        end = time.time()
        times.append(end - start)

    avg = np.mean(times)
    print(f"\nN: {n_samples:>9} | D: {n_features:>3} | K: {K:>3} | Avg: {avg:.4f}s")

    print("\n--- Centroid Stats ---")
    for i in range(K):
        print(f"Centroid {i}: {counts[i]} points")
        coords = ", ".join(f"{v:.4f}" for v in centroids[i])
        print(f"Coordinates: [ {coords} ]\n")

if __name__ == "__main__":
    run_benchmark("sample_datasets/blobs_N245760_D16_K256.bin", 245760, 16, 256)