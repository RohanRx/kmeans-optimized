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

#define BATCH_SIZE 64

__global__ void kmeans_assignment_kernel(
    float* points, 
    float* centroids, 
    float* accumulators, 
    int* counters,
    int total_points) 
{
    // Centroids remain in shared memory for fast broadcast access
    __shared__ float s_centroids[NUM_CENTROIDS * DIMS];
    __shared__ float s_accumulators[NUM_CENTROIDS * DIMS];

    int l_counters[NUM_CENTROIDS] = {0};
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Initialize shared memory
    for (int i = tid; i < NUM_CENTROIDS * DIMS; i += blockDim.x) {
        s_accumulators[i] = 0.0f;
        s_centroids[i] = centroids[i];
    }
    __syncthreads();

    for (int p = global_tid; p < total_points; p += stride) {
        // Registers to hold partial distances for each centroid
        float dists[NUM_CENTROIDS] = {0.0f};
        
        // Two sets of registers for double-buffering point data
        float reg_A[BATCH_SIZE];
        float reg_B[BATCH_SIZE];

        // 1. Initial Load: Prime the first buffer (reg_A)
        #pragma unroll
        for (int d = 0; d < BATCH_SIZE; ++d) {
            reg_A[d] = points[d * total_points + p];
        }

        // 2. Ping-Pong Pipelined Loop (Unrolled by 2)
        for (int b = 0; b < DIMS; b += 2 * BATCH_SIZE) {
            
            // --- PHASE 1: Compute on reg_A, Load into reg_B ---
            int next_b_idx = b + BATCH_SIZE;
            if (next_b_idx < DIMS) {
                #pragma unroll
                for (int d = 0; d < BATCH_SIZE; ++d) {
                    reg_B[d] = points[(next_b_idx + d) * total_points + p];
                }
            }

            // Compute distances for current batch (reg_A)
            #pragma unroll
            for (int d = 0; d < BATCH_SIZE; ++d) {
                int dim_idx = b + d;
                if (dim_idx < DIMS) { // Safety check for non-multiples of BATCH_SIZE
                    float p_val = reg_A[d];
                    for (int c = 0; c < NUM_CENTROIDS; ++c) {
                        float c_val = s_centroids[dim_idx * NUM_CENTROIDS + c];
                        float diff = p_val - c_val;
                        dists[c] += diff * diff;
                    }
                }
            }

            // --- PHASE 2: Compute on reg_B, Load into reg_A ---
            int next_a_idx = b + 2 * BATCH_SIZE;
            if (next_a_idx < DIMS) {
                #pragma unroll
                for (int d = 0; d < BATCH_SIZE; ++d) {
                    reg_A[d] = points[(next_a_idx + d) * total_points + p];
                }
            }

            // Compute distances for next batch (reg_B)
            #pragma unroll
            for (int d = 0; d < BATCH_SIZE; ++d) {
                int dim_idx = next_b_idx + d;
                if (dim_idx < DIMS) { // Safety check for non-multiples of BATCH_SIZE
                    float p_val = reg_B[d];
                    for (int c = 0; c < NUM_CENTROIDS; ++c) {
                        float c_val = s_centroids[dim_idx * NUM_CENTROIDS + c];
                        float diff = p_val - c_val;
                        dists[c] += diff * diff;
                    }
                }
            }
        }

        // 3. Find best centroid
        float min_dist = FLT_MAX;
        int best_centroid = 0;
        for (int c = 0; c < NUM_CENTROIDS; ++c) {
            if (dists[c] < min_dist) {
                min_dist = dists[c];
                best_centroid = c;
            }
        }

        l_counters[best_centroid]++;

        // 4. Update shared accumulators 
        // We re-read points from global memory here. Because we just traversed 
        // this exact memory, it will be served highly efficiently from the L1/L2 cache.
        for (int d = 0; d < DIMS; ++d) {
            float val = points[d * total_points + p];
            atomicAdd(&s_accumulators[d * NUM_CENTROIDS + best_centroid], val);
        }
    }

    __syncthreads();

    // Cooperative flush to global memory (Accumulators)
    for (int i = tid; i < NUM_CENTROIDS * DIMS; i += blockDim.x) {
        if (s_accumulators[i] != 0.0f) {
            atomicAdd(&accumulators[i], s_accumulators[i]);
        }
    }

    // Warp shuffle reduction for Counters
    int lane = tid % 32; 
    for (int c = 0; c < NUM_CENTROIDS; ++c) {
        int count = l_counters[c];
        for (int offset = 16; offset > 0; offset /= 2) {
            count += __shfl_down_sync(0xFFFFFFFF, count, offset);
        }
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
    int threads_per_block = 384;
    
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

    //Verify some output (Optional)
    std::cout << "\n--- Verification ---\n";
    for (int i = 0; i < NUM_CENTROIDS; ++i) {
        std::cout << "Points assigned to Centroid " << i << ":" << h_counters[i] << "\n";
        std::cout << "Centroid" << i << "Coordinates: [ ";
        for (int d = 0; d < 5; ++d) {
            std::cout << h_centroids[d * NUM_CENTROIDS + i];
        }
        std::cout << " ]\n"; 
    } 

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
