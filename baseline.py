import time
import numpy as np
from sklearn.cluster import KMeans
import os
import csv
import argparse

def load_bin(filename, n_samples, n_features):
    return np.fromfile(filename, dtype=np.float32).reshape(n_samples, n_features)

def run_warmup(d, k):
    X_warmup = np.random.rand(10000, d).astype(np.float32)
    
    warmup_model = KMeans(
        n_clusters=k,
        init='random',
        n_init=1,
        max_iter=10,
        algorithm='full',
        random_state=42
    )
    
    warmup_model.fit(X_warmup)
    print("Warmup complete!")

def run_benchmark(category, filename, n_samples, n_features, K, iterations):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return None

    X = load_bin(filename, n_samples, n_features)
    
    # algorithm='full' for Scikit-Learn 0.24.2 / Python 3.6.8
    model = KMeans(
        n_clusters=K,
        init=X[:K],
        n_init=1,
        max_iter=iterations,
        algorithm='full',
        random_state=42
    )

    start = time.time()
    model.fit(X)
    end = time.time()
    
    total_time = end - start
    iters = model.n_iter_
    ms_per_iter = (total_time / iters) * 1000 if iters > 0 else 0.0

    print(f"[{category.upper()}] N: {n_samples:>8} | Iters: {iters:>3} | Time: {total_time:.4f}s | Latency: {ms_per_iter:.2f} ms/iter")

    return {
        "category": category,
        "n": n_samples,
        "d": n_features,
        "k": K,
        "iters": iters,
        "total_time_s": total_time,
        "ms_per_iter": ms_per_iter
    }

def main():
    parser = argparse.ArgumentParser(description="Run CPU KMeans Benchmarks")
    parser.add_argument("-d", "--dimensions", type=int, choices=[16, 512], default=16, help="Dimensions: 16 (K=256) or 512 (K=5)")
    args = parser.parse_args()

    D = args.dimensions
    K = 256 if D == 16 else 5
    ITERATIONS = 1000
    CATEGORIES = ["easy", "hard"]
    
    BASE_DATA_DIR = os.path.join("data", f"D{D}")
    OUTPUT_DIR = os.path.join("output", f"D{D}")
    CSV_OUTPUT = os.path.join(OUTPUT_DIR, "cpu_baseline.csv")

    run_warmup(D, K)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    with open(CSV_OUTPUT, mode='w') as csv_file:
        fieldnames = ['category', 'n', 'd', 'k', 'iters', 'total_time_s', 'ms_per_iter']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cat in CATEGORIES:
            print(f"\n--- Benchmarking D{D} {cat.upper()} Datasets ---")
            folder = os.path.join(BASE_DATA_DIR, cat)
            
            if not os.path.exists(folder):
                print(f"Directory not found: {folder}")
                continue

            # Sort files numerically by N
            files = sorted([f for f in os.listdir(folder) if f.endswith('.bin')], 
                           key=lambda x: int(x.split('_')[1][1:]))
            
            for f in files:
                try:
                    parts = f.split('_')
                    N = int(parts[1][1:]) 
                    path = os.path.join(folder, f)
                    
                    res = run_benchmark(cat, path, N, D, K, ITERATIONS)
                    if res:
                        writer.writerow(res)
                        csv_file.flush()
                        
                except Exception as e:
                    print(f"Skipping {f}: {e}")

    print(f"\n[✓] All results saved to: {CSV_OUTPUT}")

if __name__ == "__main__":
    main()