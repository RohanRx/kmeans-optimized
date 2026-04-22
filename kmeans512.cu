#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <float.h>
#include <fstream>
#include <sstream>
#include <string>

#define DIMS 512
#define NUM_CENTROIDS 5

__global__ void kmeans_assignment_kernel(
    float* points, 
    float* centroids, 
    float* accumulators, 
    int* counters,
    int total_points) 
{
    __shared__ float s_centroids[NUM_CENTROIDS * DIMS];
    __shared__ float s_accumulators[NUM_CENTROIDS * DIMS];

    int l_counters[NUM_CENTROIDS] = {0};

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < NUM_CENTROIDS * DIMS; i += blockDim.x) {
        s_accumulators[i] = 0.0f;
        s_centroids[i] = centroids[i];
    }

    __syncthreads();

    for (int p = global_tid; p < total_points; p += stride) {
        float min_dist = FLT_MAX;
        int best_centroid = 0;

        // Calculate distance to all centroids
        for (int c = 0; c < NUM_CENTROIDS; ++c) {
            float dist = 0.0f;
            for (int d = 0; d < DIMS; ++d) {
                // Accessing SoA layout: index = dimension * total_elements + element_index
                float diff = points[d * total_points + p] - s_centroids[d * NUM_CENTROIDS + c];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_centroid = c;
            }
        }

        l_counters[best_centroid]++;

        // Accumulate point dimensions into shared memory using atomics
        for (int d = 0; d < DIMS; ++d) {
            float val = points[d * total_points + p];
            atomicAdd(&s_accumulators[d * NUM_CENTROIDS + best_centroid], val);
        }
    }

    // Wait for all threads in the block to finish accumulating
    __syncthreads();

    // 3. Flush shared accumulators back to global memory cooperatively
    for (int i = tid; i < NUM_CENTROIDS * DIMS; i += blockDim.x) {
        if (s_accumulators[i] != 0.0f) {
            atomicAdd(&accumulators[i], s_accumulators[i]);
        }
    }

    int lane = tid % 32; 
    
    for (int c = 0; c < NUM_CENTROIDS; ++c) {
        int count = l_counters[c];
        
        // Shuffle down sync to accumulate counts across the warp
        for (int offset = 16; offset > 0; offset /= 2) {
            count += __shfl_down_sync(0xFFFFFFFF, count, offset);
        }

        // Only lane 0 of each warp writes the aggregated sum to global memory
        if (lane == 0 && count > 0) {
            atomicAdd(&counters[c], count);
        }
    }
}

__global__ void kmeans_update_kernel(float* centroids, float* accumulators, int* counters, int iter, int total_iters) {
    int tid = threadIdx.x;
    int total_elements = NUM_CENTROIDS * DIMS;

    // 1. Block-stride loop: 1024 threads will process all 2560 elements
    for (int idx = tid; idx < total_elements; idx += blockDim.x) {
        // Extract the centroid ID by taking the modulo
        int c_idx = idx % NUM_CENTROIDS;
        int count = counters[c_idx];

        // Guard against division by zero
        if (count > 0) {
            centroids[idx] = accumulators[idx] / (float)count;
            accumulators[idx] = 0.0f;
        }
    }
    __syncthreads();

    if (iter < total_iters - 1) {
        if (tid < NUM_CENTROIDS) {
            counters[tid] = 0;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <filename> <iters>\n";
        return 1;
    }

    int num_blocks = 1024;
    int threads_per_block = 256;
    
    std::string filename = argv[1];
    int ITERS = std::stoi(argv[2]);
    
    // 1. Open the file in binary mode and start at the end to get file size
    std::ifstream file(filename, std::ios::binary | std::ios::ate);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << "\n";
        return 1;
    }

    // Calculate sizes based on the binary file byte footprint
    std::streamsize file_size = file.tellg();
    file.seekg(0, std::ios::beg); // Rewind back to the beginning

    size_t total_floats = file_size / sizeof(float);
    int total_points = total_floats / DIMS;

    std::cout << "Reading binary dataset...\n";

    // 2. Read the entire raw binary dump directly into our AoS vector
    std::vector<float> temp_aos_points(total_floats);
    if (!file.read(reinterpret_cast<char*>(temp_aos_points.data()), file_size)) {
        std::cerr << "Error: Failed to read binary data completely.\n";
        return 1;
    }
    file.close();

    std::cout << "Successfully loaded " << total_points << " points.\n";

    // Memory sizes
    size_t points_bytes = total_points * DIMS * sizeof(float);
    size_t centroids_bytes = NUM_CENTROIDS * DIMS * sizeof(float);
    size_t accumulators_bytes = NUM_CENTROIDS * DIMS * sizeof(float);
    size_t counters_bytes = NUM_CENTROIDS * sizeof(int);

    float* h_points = (float*)malloc(points_bytes);
    float* h_centroids = (float*)malloc(centroids_bytes);
    float* h_accumulators = (float*)malloc(accumulators_bytes);
    int* h_counters = (int*)malloc(counters_bytes);

    // 3. Convert AoS (Binary layout) to SoA (Kernel layout) directly
    for (int p = 0; p < total_points; ++p) {
        for (int d = 0; d < DIMS; ++d) {
            h_points[d * total_points + p] = temp_aos_points[p * DIMS + d];
        }
    }

    // 4. Initialize Centroids (Picking the first NUM_CENTROIDS points from the dataset)
    for (int c = 0; c < NUM_CENTROIDS; ++c) {
        for (int d = 0; d < DIMS; ++d) {
            h_centroids[d * NUM_CENTROIDS + c] = temp_aos_points[c * DIMS + d];
        }
    }

    memset(h_accumulators, 0, accumulators_bytes);
    memset(h_counters, 0, counters_bytes);

    float *d_points, *d_centroids, *d_accumulators;
    int *d_counters;

    // initialize device memory
    cudaMalloc(&d_points, points_bytes);
    cudaMalloc(&d_centroids, centroids_bytes);
    cudaMalloc(&d_accumulators, accumulators_bytes);
    cudaMalloc(&d_counters, counters_bytes);

    // copy data to device
    cudaMemcpy(d_points, h_points, points_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids, centroids_bytes, cudaMemcpyHostToDevice);
    
    // Initialize device accumulators and counters to 0
    cudaMemset(d_accumulators, 0, accumulators_bytes);
    cudaMemset(d_counters, 0, counters_bytes);

    // CUDA events for timing the kernel execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    
    for (int iter = 0; iter < ITERS; ++iter) {
        kmeans_assignment_kernel<<<num_blocks, threads_per_block>>>(
            d_points, 
            d_centroids, 
            d_accumulators, 
            d_counters, 
            total_points
        );
        cudaDeviceSynchronize();
        kmeans_update_kernel<<<1, 1024>>>(
            d_centroids, 
            d_accumulators, 
            d_counters,
            iter,
            ITERS
        );
        cudaDeviceSynchronize();
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel Execution Time: " << milliseconds << " ms\n";
    std::cout << "Time Per Run: " << milliseconds/ITERS << " ms\n";

    // copy data back
    cudaMemcpy(h_centroids, d_centroids, centroids_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counters, d_counters, counters_bytes, cudaMemcpyDeviceToHost);

    // Verify some output (Optional)
    // std::cout << "\n--- Verification ---\n";
    // for (int i = 0; i < NUM_CENTROIDS; ++i) {
    //     std::cout << "Points assigned to Centroid " << i << ":" << h_counters[i] << "\n";
    
    //     std::cout << "Centroid" << i << "Coordinates: [ ";
    //     for (int d = 0; d < DIMS; ++d) {
    //         // Because of SoA layout, dimension 'd' for centroid '0' is at index (d * NUM_CENTROIDS + 0)
    //         std::cout << h_centroids[d * NUM_CENTROIDS + i];
            
    //         if (d < DIMS - 1) std::cout << ", ";
    //     }
    //     std::cout << " ]\n"; 
    // } 

    // Free device memory
    cudaFree(d_points);
    cudaFree(d_centroids);
    cudaFree(d_accumulators);
    cudaFree(d_counters);

    // Free host memory
    free(h_points);
    free(h_centroids);
    free(h_accumulators);
    free(h_counters);

    return 0;
}
