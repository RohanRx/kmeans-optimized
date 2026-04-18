import time
import numpy as np
from sklearn.cluster import KMeans
import os
import csv

BASE_DATA_DIR = "data_D16"
CATEGORIES = ["easy", "hard"]
OUTPUT_DIR = "output"
CSV_OUTPUT = os.path.join(OUTPUT_DIR, "cpu_baseline.csv")

ITERATIONS = 1000

def load_bin(filename, n_samples, n_features):
    return np.fromfile(filename, dtype=np.float32).reshape(n_samples, n_features)

def run_benchmark(category, filename, n_samples, n_features, K):
    if not os.path.exists(filename):
        print("File not found: {}".format(filename))
        return None

    X = load_bin(filename, n_samples, n_features)
    
    # algorithm='full' for Scikit-Learn 0.24.2 / Python 3.6.8
    model = KMeans(
        n_clusters=K,
        init=X[:K],
        n_init=1,
        max_iter=ITERATIONS,
        algorithm='full',
        random_state=42
    )

    start = time.time()
    model.fit(X)
    end = time.time()
    
    total_time = end - start
    iters = model.n_iter_
    ms_per_iter = (total_time / iters) * 1000

    print("[{}] N: {:>8} | Iters: {:>3} | Time: {:.4f}s | Latency: {:.2f} ms/iter".format(
          category.upper(), n_samples, iters, total_time, ms_per_iter))

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
    D = 16
    K = 256
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    with open(CSV_OUTPUT, mode='w') as csv_file:
        fieldnames = ['category', 'n', 'd', 'k', 'iters', 'total_time_s', 'ms_per_iter']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for cat in CATEGORIES:
            print("\n--- Benchmarking {} Datasets ---".format(cat.upper()))
            folder = os.path.join(BASE_DATA_DIR, cat)
            
            if not os.path.exists(folder):
                print("Directory not found: {}".format(folder))
                continue

            # Sort files numerically by N
            files = sorted([f for f in os.listdir(folder) if f.endswith('.bin')], 
                           key=lambda x: int(x.split('_')[1][1:]))
            
            for f in files:
                try:
                    parts = f.split('_')
                    N = int(parts[1][1:]) 
                    path = os.path.join(folder, f)
                    
                    res = run_benchmark(cat, path, N, D, K)
                    if res:
                        writer.writerow(res)
                        csv_file.flush()
                        
                except Exception as e:
                    print("Skipping {}: {}".format(f, e))

    print("\n[✓] All results saved to: {}".format(CSV_OUTPUT))

if __name__ == "__main__":
    main()