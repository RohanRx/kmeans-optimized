#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <float.h>
#include <fstream>
#include <sstream>
#include <string>


#define DIMS 16
#define NUM_CENTROIDS 256
#define POINTS_PER_BATCH 240
#define ITERS 50

__global__ void kmeans_assignment_kernel(
    float* points, 
    float* centroids, 
    float* accumulators, 
    int* counters,
    int total_points) 
    {

    __shared__ float s_points[POINTS_PER_BATCH * DIMS];          // 15.36 KB
    __shared__ float s_centroids[NUM_CENTROIDS * DIMS];          // 16.38 KB
    __shared__ float s_accumulators[NUM_CENTROIDS * DIMS];       // 16.38 KB
    __shared__ int s_counters[NUM_CENTROIDS];
    const int rbuff_size = 15;
    float r_buff[rbuff_size] = {0.0};
    bool is_first_batch = true;
    //float p_regs[16][5] = {0.0};
    // 1 threadblock only per SM
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;
    
    //Initialize shared memory
    for (int i = tid; i < NUM_CENTROIDS * DIMS; i += blockDim.x) {
        s_accumulators[i] = 0.0f;
        s_centroids[i] = centroids[i];
    }
    if (tid < NUM_CENTROIDS) {
        s_counters[tid] = 0;
    }
    
    __syncthreads();

    //total batches is total_points/240
    //start batch is based on block index
    int total_batches = (total_points + POINTS_PER_BATCH - 1) / POINTS_PER_BATCH;
    int batch_idx = blockIdx.x;

    while (batch_idx < total_batches) {
        //start loading batch of points into shared memory
        int start_idx = batch_idx * POINTS_PER_BATCH;
        
        if (is_first_batch) {
            // First iteration: Load directly from Global Memory -> Shared Memory
            #pragma unroll
            for (int step = 0; step < rbuff_size; ++step) {
                int i = tid + step * blockDim.x;
                if (i < POINTS_PER_BATCH * DIMS) {
                    int d = i / POINTS_PER_BATCH; 
                    int p = i % POINTS_PER_BATCH; 
                    s_points[i] = points[d * total_points + start_idx + p];
                }
            }
            is_first_batch = false;
        } else {
            // Subsequent iterations: Move prefetched data from Registers -> Shared Memory
            #pragma unroll
            for (int step = 0; step < rbuff_size; ++step) {
                int i = tid + step * blockDim.x;
                if (i < POINTS_PER_BATCH * DIMS) {
                    s_points[i] = r_buff[step]; // Uses registers directly via constant index!
                }
            }
        }

        // bring next batch into registers
        int next_batch_idx = batch_idx + gridDim.x;
        if (next_batch_idx < total_batches) {
            int next_start_idx = next_batch_idx * POINTS_PER_BATCH;
            
            #pragma unroll
            for (int step = 0; step < rbuff_size; ++step) {
                int i = tid + step * blockDim.x;
                if (i < POINTS_PER_BATCH * DIMS) {
                    int d = i / POINTS_PER_BATCH; 
                    int p = i % POINTS_PER_BATCH; 
                    r_buff[step] = points[d * total_points + next_start_idx + p];
                }
            }
        }

        // Wait for all threads to finish populating shared memory
        __syncthreads();

        int points_per_warp = POINTS_PER_BATCH/(blockDim.x/32);
        int warp_point_start = wid * points_per_warp;

        float p_regs[16][10] = {0.0};

        for (int p_base = 0; p_base < points_per_warp; p_base += 10) { 
            
            // 1. Load a batch of 10 points into registers explicitly
            int p_idx_0 = warp_point_start + p_base + 0;
            int p_idx_1 = warp_point_start + p_base + 1;
            int p_idx_2 = warp_point_start + p_base + 2;
            int p_idx_3 = warp_point_start + p_base + 3;
            int p_idx_4 = warp_point_start + p_base + 4;
            int p_idx_5 = warp_point_start + p_base + 5;
            int p_idx_6 = warp_point_start + p_base + 6;
            int p_idx_7 = warp_point_start + p_base + 7;
            int p_idx_8 = warp_point_start + p_base + 8;
            int p_idx_9 = warp_point_start + p_base + 9;

            for (int d = 0; d < DIMS; ++d) {
                p_regs[d][0] = s_points[d * POINTS_PER_BATCH + p_idx_0];
                p_regs[d][1] = s_points[d * POINTS_PER_BATCH + p_idx_1];
                p_regs[d][2] = s_points[d * POINTS_PER_BATCH + p_idx_2];
                p_regs[d][3] = s_points[d * POINTS_PER_BATCH + p_idx_3];
                p_regs[d][4] = s_points[d * POINTS_PER_BATCH + p_idx_4];
                p_regs[d][5] = s_points[d * POINTS_PER_BATCH + p_idx_5];
                p_regs[d][6] = s_points[d * POINTS_PER_BATCH + p_idx_6];
                p_regs[d][7] = s_points[d * POINTS_PER_BATCH + p_idx_7];
                p_regs[d][8] = s_points[d * POINTS_PER_BATCH + p_idx_8];
                p_regs[d][9] = s_points[d * POINTS_PER_BATCH + p_idx_9];
            }

            // 2. Initialize distance and centroid tracking for all 10 points
            float min_dist_0 = FLT_MAX; float min_dist_1 = FLT_MAX;
            float min_dist_2 = FLT_MAX; float min_dist_3 = FLT_MAX;
            float min_dist_4 = FLT_MAX; float min_dist_5 = FLT_MAX;
            float min_dist_6 = FLT_MAX; float min_dist_7 = FLT_MAX;
            float min_dist_8 = FLT_MAX; float min_dist_9 = FLT_MAX;

            int closest_centroid_0 = -1; int closest_centroid_1 = -1;
            int closest_centroid_2 = -1; int closest_centroid_3 = -1;
            int closest_centroid_4 = -1; int closest_centroid_5 = -1;
            int closest_centroid_6 = -1; int closest_centroid_7 = -1;
            int closest_centroid_8 = -1; int closest_centroid_9 = -1;

            // 3. Loop centroids (all threads in warp get different centroids)
            for (int c = 0; c < 8; ++c) { 
                int c_idx = lane + c * 32; 
                
                float dist_0 = 0.0f; float dist_1 = 0.0f;
                float dist_2 = 0.0f; float dist_3 = 0.0f;
                float dist_4 = 0.0f; float dist_5 = 0.0f;
                float dist_6 = 0.0f; float dist_7 = 0.0f;
                float dist_8 = 0.0f; float dist_9 = 0.0f;

                for (int d = 0; d < DIMS; ++d) { 
                    // Load the centroid dimension once, reuse it 10 times
                    float c_val = s_centroids[d * NUM_CENTROIDS + c_idx]; 
                    
                    float diff_0 = p_regs[d][0] - c_val;
                    float diff_1 = p_regs[d][1] - c_val;
                    float diff_2 = p_regs[d][2] - c_val;
                    float diff_3 = p_regs[d][3] - c_val;
                    float diff_4 = p_regs[d][4] - c_val;
                    float diff_5 = p_regs[d][5] - c_val;
                    float diff_6 = p_regs[d][6] - c_val;
                    float diff_7 = p_regs[d][7] - c_val;
                    float diff_8 = p_regs[d][8] - c_val;
                    float diff_9 = p_regs[d][9] - c_val;

                    dist_0 += diff_0 * diff_0;
                    dist_1 += diff_1 * diff_1;
                    dist_2 += diff_2 * diff_2;
                    dist_3 += diff_3 * diff_3;
                    dist_4 += diff_4 * diff_4;
                    dist_5 += diff_5 * diff_5;
                    dist_6 += diff_6 * diff_6;
                    dist_7 += diff_7 * diff_7;
                    dist_8 += diff_8 * diff_8;
                    dist_9 += diff_9 * diff_9;
                }

                // Update local minimums for the 10 points
                if (dist_0 < min_dist_0) { min_dist_0 = dist_0; closest_centroid_0 = c_idx; }
                if (dist_1 < min_dist_1) { min_dist_1 = dist_1; closest_centroid_1 = c_idx; }
                if (dist_2 < min_dist_2) { min_dist_2 = dist_2; closest_centroid_2 = c_idx; }
                if (dist_3 < min_dist_3) { min_dist_3 = dist_3; closest_centroid_3 = c_idx; }
                if (dist_4 < min_dist_4) { min_dist_4 = dist_4; closest_centroid_4 = c_idx; }
                if (dist_5 < min_dist_5) { min_dist_5 = dist_5; closest_centroid_5 = c_idx; }
                if (dist_6 < min_dist_6) { min_dist_6 = dist_6; closest_centroid_6 = c_idx; }
                if (dist_7 < min_dist_7) { min_dist_7 = dist_7; closest_centroid_7 = c_idx; }
                if (dist_8 < min_dist_8) { min_dist_8 = dist_8; closest_centroid_8 = c_idx; }
                if (dist_9 < min_dist_9) { min_dist_9 = dist_9; closest_centroid_9 = c_idx; }
            }

            // 4. Warp reduction to find the global closest centroid for all 10 points
            for (int offset = 16; offset > 0; offset /= 2) {
                float other_dist_0 = __shfl_down_sync(0xFFFFFFFF, min_dist_0, offset);
                int other_centroid_0 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_0, offset);
                if (other_dist_0 < min_dist_0) { min_dist_0 = other_dist_0; closest_centroid_0 = other_centroid_0; }

                float other_dist_1 = __shfl_down_sync(0xFFFFFFFF, min_dist_1, offset);
                int other_centroid_1 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_1, offset);
                if (other_dist_1 < min_dist_1) { min_dist_1 = other_dist_1; closest_centroid_1 = other_centroid_1; }

                float other_dist_2 = __shfl_down_sync(0xFFFFFFFF, min_dist_2, offset);
                int other_centroid_2 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_2, offset);
                if (other_dist_2 < min_dist_2) { min_dist_2 = other_dist_2; closest_centroid_2 = other_centroid_2; }

                float other_dist_3 = __shfl_down_sync(0xFFFFFFFF, min_dist_3, offset);
                int other_centroid_3 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_3, offset);
                if (other_dist_3 < min_dist_3) { min_dist_3 = other_dist_3; closest_centroid_3 = other_centroid_3; }

                float other_dist_4 = __shfl_down_sync(0xFFFFFFFF, min_dist_4, offset);
                int other_centroid_4 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_4, offset);
                if (other_dist_4 < min_dist_4) { min_dist_4 = other_dist_4; closest_centroid_4 = other_centroid_4; }

                float other_dist_5 = __shfl_down_sync(0xFFFFFFFF, min_dist_5, offset);
                int other_centroid_5 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_5, offset);
                if (other_dist_5 < min_dist_5) { min_dist_5 = other_dist_5; closest_centroid_5 = other_centroid_5; }

                float other_dist_6 = __shfl_down_sync(0xFFFFFFFF, min_dist_6, offset);
                int other_centroid_6 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_6, offset);
                if (other_dist_6 < min_dist_6) { min_dist_6 = other_dist_6; closest_centroid_6 = other_centroid_6; }

                float other_dist_7 = __shfl_down_sync(0xFFFFFFFF, min_dist_7, offset);
                int other_centroid_7 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_7, offset);
                if (other_dist_7 < min_dist_7) { min_dist_7 = other_dist_7; closest_centroid_7 = other_centroid_7; }

                float other_dist_8 = __shfl_down_sync(0xFFFFFFFF, min_dist_8, offset);
                int other_centroid_8 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_8, offset);
                if (other_dist_8 < min_dist_8) { min_dist_8 = other_dist_8; closest_centroid_8 = other_centroid_8; }

                float other_dist_9 = __shfl_down_sync(0xFFFFFFFF, min_dist_9, offset);
                int other_centroid_9 = __shfl_down_sync(0xFFFFFFFF, closest_centroid_9, offset);
                if (other_dist_9 < min_dist_9) { min_dist_9 = other_dist_9; closest_centroid_9 = other_centroid_9; }
            }

            // 5. Atomically add to accumulator (lane 0 only)
            if (lane == 0) {
                atomicAdd(&s_counters[closest_centroid_0], 1);
                atomicAdd(&s_counters[closest_centroid_1], 1);
                atomicAdd(&s_counters[closest_centroid_2], 1);
                atomicAdd(&s_counters[closest_centroid_3], 1);
                atomicAdd(&s_counters[closest_centroid_4], 1);
                atomicAdd(&s_counters[closest_centroid_5], 1);
                atomicAdd(&s_counters[closest_centroid_6], 1);
                atomicAdd(&s_counters[closest_centroid_7], 1);
                atomicAdd(&s_counters[closest_centroid_8], 1);
                atomicAdd(&s_counters[closest_centroid_9], 1);
                
                for (int d = 0; d < DIMS; ++d) {
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_0], p_regs[d][0]);
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_1], p_regs[d][1]);
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_2], p_regs[d][2]);
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_3], p_regs[d][3]);
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_4], p_regs[d][4]);
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_5], p_regs[d][5]);
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_6], p_regs[d][6]);
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_7], p_regs[d][7]);
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_8], p_regs[d][8]);
                    atomicAdd(&s_accumulators[d * NUM_CENTROIDS + closest_centroid_9], p_regs[d][9]);
                }
            }
        }
        __syncthreads();
        batch_idx += gridDim.x;
    }
    // all batches are done from this block, now we can globally flush to main memory
    for (int i = tid; i < NUM_CENTROIDS * DIMS; i += blockDim.x) {
        // Find which centroid this specific dimension belongs to
        int c_idx = i % NUM_CENTROIDS; 
        // Only do the atomicAdd if this centroid actually received points
        if (s_counters[c_idx] > 0) {
            atomicAdd(&accumulators[i], s_accumulators[i]);
        }
    }
    
    if (tid < NUM_CENTROIDS) {
        if (s_counters[tid] > 0) {
            atomicAdd(&counters[tid], s_counters[tid]);
        }
    }
}

__global__ void kmeans_update_kernel(float* centroids, float* accumulators, int* counters, int iter) {
    // 1 thread handles 1 centroid completely
    int c_idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (c_idx < NUM_CENTROIDS) {
        int count = counters[c_idx];

        // Guard against division by zero (if a cluster became empty)
        if (count > 0) {
            for (int d = 0; d < DIMS; ++d) {
                // Calculate SoA index
                int idx = d * NUM_CENTROIDS + c_idx;
                //Calculate the mean and set the new centroid position
                centroids[idx] = accumulators[idx] / (float)count;
                //Reset the accumulator to 0 for the next iteration
                accumulators[idx] = 0.0f;
            }
            //Reset the counter to 0 for the next iteration
            if (iter < ITERS-1) {
                counters[c_idx] = 0;
            }
        }
    }
}

int main() {
    int num_blocks = 1024;
    int threads_per_block = 256;
    
    // Change this to your actual binary dataset filename
    std::string filename = "sample_datasets/blobs_N245760_D16_K256.bin"; 
    
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
        kmeans_update_kernel<<<1, 256>>>(
            d_centroids, 
            d_accumulators, 
            d_counters,
            iter
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
    std::cout << "\n--- Verification ---\n";
    for (int i = 0; i < 2; ++i) {
        std::cout << "Points assigned to Centroid" << i << ":" << h_counters[i] << "\n";
    
        std::cout << "Centroid" << i << "Coordinates: [ ";
        for (int d = 0; d < DIMS; ++d) {
            // Because of SoA layout, dimension 'd' for centroid '0' is at index (d * NUM_CENTROIDS + 0)
            std::cout << h_centroids[d * NUM_CENTROIDS + i];
            
            if (d < DIMS - 1) std::cout << ", ";
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
