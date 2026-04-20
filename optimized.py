import subprocess
import os
import csv
import time
import pandas as pd

BASE_DATA_DIR = "data_D16"
CATEGORIES = ["easy", "hard"]
OUTPUT_DIR = "output"
CSV_INPUT = os.path.join(OUTPUT_DIR, "cpu_baseline.csv")
GPU_CSV_OUTPUT = os.path.join(OUTPUT_DIR, "gpu_optimized.csv")

CUDA_SRC = "kmeans.cu"
CUDA_EXE = "./kmeans_optimized"

def compile_cuda():
    print("Compiling {}...".format(CUDA_SRC))
    # Using sm_75 for Tesla T4 optimization
    cmd = ["nvcc", "-O3", "-arch=sm_75", CUDA_SRC, "-o", CUDA_EXE]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("Compilation Failed!\n{}".format(result.stderr))
        exit(1)
    print("Compilation successful.")

def run_gpu_benchmark(category, filename, n_samples, d_features, K, iters):
    # Pass filename and iters as command line arguments to the CUDA binary
    cmd = [CUDA_EXE, filename, str(iters)]
    
    try:
        start_wall = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        end_wall = time.time()
        
        if result.returncode != 0:
            print("Error running {}: {}".format(filename, result.stderr))
            return None

        # Parse the custom CUDA output
        output = result.stdout
        ms_per_iter = 0.0
        for line in output.split('\n'):
            if "Time Per Run:" in line:
                # Extract number from "Time Per Run: 12.34 ms"
                ms_per_iter = float(line.split(':')[1].strip().split(' ')[0])

        total_time = end_wall - start_wall

        print("[{}] N: {:>8} | Iters: {:>3} | Time: {:.4f}s | Latency: {:.2f} ms/iter".format(
          category.upper(), n_samples, iters, total_time, ms_per_iter))

        return {
            "category": category,
            "n": n_samples,
            "d": d_features,
            "k": K,
            "iters": iters,
            "total_time_s": total_time,
            "ms_per_iter": ms_per_iter
        }
    except Exception as e:
        print("Failed to run {}: {}".format(filename, e))
        return None

def main():
    if not os.path.exists(CSV_INPUT):
        print("Baseline CSV not found at {}. Run baseline first.".format(CSV_INPUT))
        return

    compile_cuda()
    df = pd.read_csv(CSV_INPUT)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    with open(GPU_CSV_OUTPUT, mode='w', newline='') as csv_file:
        fieldnames = ['category', 'n', 'd', 'k', 'iters', 'total_time_s', 'ms_per_iter']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        print("\n--- Benchmarking GPU Optimized Datasets ---")

        for _, row in df.iterrows():
            cat = row['category']
            N = int(row['n'])
            D = int(row['d'])
            K = int(row['k'])
            ITERS = int(row['iters'])
            
            filename = os.path.join(BASE_DATA_DIR, cat, "blobs_N{}_D{}_K{}.bin".format(N, D, K))
            
            res = run_gpu_benchmark(cat, filename, N, D, K, ITERS)
            
            if res:
                writer.writerow(res)
                csv_file.flush()

    print("\n[✓] GPU results saved to: {}".format(GPU_CSV_OUTPUT))

if __name__ == "__main__":
    main()