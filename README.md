# CUDA K-Means Optimization

This project implements a CUDA-accelerated K-Means clustering algorithm optimized for the NVIDIA Tesla T4. We developed two specialized kernels:
one for low-dimensional data (D=16, K=256) and one for high-dimensional data
(D=512, K=5). Against a scikit-learn CPU baseline, we achieve average speedups of 17.34x and 9.54x for D=16 and D=512 respectively.

The intended workflow is:

1. Generate datasets with `create_dataset.py`
2. Run the CPU baseline with `baseline.py`
3. Run the CUDA benchmark with `optimized.py`
4. Compare both outputs with `analyze_results.py`

The scripts are designed to be run for one dimensionality at a time:

- `-d 16`
- `-d 512`

## File Structure

```text
kmeans-optimized/
├── README.md
├── requirements.txt
├── Makefile
├── create_dataset.py          # Generates easy/hard make_blobs datasets
├── baseline.py                # CPU scikit-learn KMeans benchmark
├── optimized.py               # Compiles and runs the CUDA benchmark
├── analyze_results.py         # Computes speedup CSV, plot, and summary stats
├── kmeans16.cu                # CUDA kernel for D=16, K=256
├── kmeans512.cu               # CUDA kernel for D=512, K=5
├── helper/
│   ├── plot_results.py        # Plots any benchmark CSV in output/D*/
│   ├── accuracy_test.py       # Accuracy check for the CUDA-style Lloyd's implementation
│   └── plot_params.py         # Compares CSVs for different threads-per-block settings
├── data/                      # Generated datasets (created after running create_dataset.py)
├── output/                    # Benchmark CSVs, plots, and speedup analysis
└── sample_datasets/           # Example binary datasets
```

## Requirements

You need:

- Python 3
- NVIDIA CUDA toolkit with `nvcc`
- A CUDA-capable GPU
- Python packages from `requirements.txt`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Current Python dependencies in the repo:

- `numpy==1.19.5`
- `scikit-learn==0.24.2`
- `matplotlib==3.3.4`
- `pandas`

## Important Notes

- Run all commands from the `kmeans-optimized/` directory.
- The workflow is sequential: dataset generation -> CPU baseline -> GPU benchmark -> analysis.
- Each script expects either `-d 16` or `-d 512`.
- `optimized.py` compiles with `nvcc -O3 -arch=sm_75`, so if your GPU architecture is different you may need to update that flag.
- For `D=16`, the code uses `K=256`.
- For `D=512`, the code uses `K=5`.

## End-to-End Workflow

### 1. Generate datasets

Generate all benchmark datasets for one dimensionality:

```bash
python create_dataset.py -d 16
```

or

```bash
python create_dataset.py -d 512
```

What it does:

- Creates datasets with `sklearn.datasets.make_blobs`
- Writes both `easy` and `hard` datasets
- Stores them under `data/D16/...` or `data/D512/...`
- Doubles `N` each step starting at `3840`
- Stops when a dataset would exceed ~2 GB

Generated layout:

```text
data/
└── D16 or D512
    ├── easy/
    │   └── blobs_N*_D*_K*.bin
    └── hard/
        └── blobs_N*_D*_K*.bin
```

### 2. Run the CPU baseline

After datasets exist, run:

```bash
python baseline.py -d 16
```

or

```bash
python baseline.py -d 512
```

What it does:

- Loads every generated dataset for the selected dimensionality
- Runs scikit-learn `KMeans`
- Measures total runtime and computes `ms_per_iter`
- Writes results to:

```text
output/D16/cpu_baseline.csv
```

or

```text
output/D512/cpu_baseline.csv
```

### 3. Run the CUDA benchmark

After the CPU CSV exists, run:

```bash
python optimized.py -d 16
```

or

```bash
python optimized.py -d 512
```

What it does:

- Compiles `kmeans16.cu` or `kmeans512.cu`
- Reads the iteration counts from `cpu_baseline.csv`
- Runs the CUDA executable on the same datasets
- Writes GPU benchmark results to:

```text
output/D16/gpu_optimized.csv
```

or

```text
output/D512/gpu_optimized.csv
```

Optional suffix:

```bash
python optimized.py -d 16 -e _256
```

This produces:

```text
output/D16/gpu_optimized_256.csv
```

That is useful for block-size ablation studies.

### 4. Analyze speedup

After both CPU and GPU CSVs exist, run:

```bash
python analyze_results.py -d 16
```

or

```bash
python analyze_results.py -d 512
```

What it does:

- Merges CPU and GPU benchmark CSVs
- Computes speedup as:

```text
speedup = ms_per_iter_cpu / ms_per_iter_gpu
```

- Writes a speedup table CSV
- Saves a speedup plot
- Prints summary statistics to the terminal

Outputs:

```text
output/D16/speedup_results.csv
output/D16/plot_speedup_analysis.png
```

or

```text
output/D512/speedup_results.csv
output/D512/plot_speedup_analysis.png
```

Optional suffix support matches `optimized.py`:

```bash
python analyze_results.py -d 16 -e _256
```

## Recommended Command Sequences

### D = 16

```bash
python create_dataset.py -d 16
python baseline.py -d 16
python optimized.py -d 16
python analyze_results.py -d 16
```

### D = 512

```bash
python create_dataset.py -d 512
python baseline.py -d 512
python optimized.py -d 512
python analyze_results.py -d 512
```

## Output Files

For each dimensionality, the main outputs are:

- `output/D*/cpu_baseline.csv`
- `output/D*/gpu_optimized.csv`
- `output/D*/speedup_results.csv`
- `output/D*/plot_speedup_analysis.png`

The benchmark CSVs contain:

- `category`
- `n`
- `d`
- `k`
- `iters`
- `total_time_s`
- `ms_per_iter`

The speedup CSV contains:

- `category`
- `n`
- `iters`
- `ms_per_iter_cpu`
- `ms_per_iter_gpu`
- `speedup`

## Helper Scripts

### `helper/plot_results.py`

This script plots any benchmark CSV produced by `baseline.py` or `optimized.py`.

Example:

```bash
python helper/plot_results.py -d 16 -f cpu_baseline.csv
python helper/plot_results.py -d 16 -f gpu_optimized.csv
```

It saves:

```text
output/D16/plot_cpu_baseline.png
output/D16/plot_gpu_optimized.png
```

or the corresponding `D512` versions.

### `helper/accuracy_test.py`

This script is for accuracy testing of the CUDA-style Lloyd's K-means logic against a fixed dataset setup.

Run it from the project root:

```bash
python helper/accuracy_test.py
```

Notes:

- It currently uses a hardcoded dataset path: `data/D512/easy/blobs_N3840_D512_K5.bin`
- It prints centroid and cluster-count information
- It is meant as a correctness/debugging utility, not the main benchmark pipeline

### `helper/plot_params.py`

This script compares multiple GPU CSV outputs, such as different threads-per-block experiments.

It looks for files like:

```text
output/D16/block_test/gpu_optimized_256.csv
output/D16/block_test/gpu_optimized_384.csv
```

and generates comparison plots for `easy` and `hard`.

Run:

```bash
python helper/plot_params.py
```

Notes:

- The script currently uses a fixed `DIMENSIONS = 16`
- To compare `D512` results, update that constant in the script first

## CUDA Build Notes

`optimized.py` handles compilation automatically, but there is also a simple `Makefile`:

```bash
make
```

This builds:

- `kmeans16.x`
- `kmeans512.x`

Clean build artifacts with:

```bash
make clean
```

## Typical Directory Outputs After Running

```text
output/
├── D16/
│   ├── cpu_baseline.csv
│   ├── gpu_optimized.csv
│   ├── speedup_results.csv
│   ├── plot_speedup_analysis.png
│   └── block_test/
└── D512/
    ├── cpu_baseline.csv
    ├── gpu_optimized.csv
    ├── speedup_results.csv
    ├── plot_speedup_analysis.png
    └── block_test/
```

## Summary

If you only want the main workflow, use this order:

```bash
python create_dataset.py -d 16
python baseline.py -d 16
python optimized.py -d 16
python analyze_results.py -d 16
```

or replace `16` with `512`.
