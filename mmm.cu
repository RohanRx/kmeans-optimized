#include <iostream>

using namespace std;

__global__ void kernel_call(float *c) {

    __shared__ float buffer[12 * 1024];

    float *s_c = buffer;
    float *s_a = buffer + 4096;
    float *s_b = buffer + 4096 * 2;

    // 1 threadblock only
    int id = threadIdx.x;
    int p = blockDim.x;

    /**** Do Not Change Code Above This ****/
    int warp_id = id / 32;
    int lane_id = id % 32;

    int chunk_row = lane_id / 8; // 32 threads / 8 cols = 4 rows (ranges 0-3)
    int chunk_col = lane_id % 8; // ranges 0-7

    int chunk_grid_row = warp_id / 4; // 32 warps / 4 cols = 8 rows (ranges 0-7)
    int chunk_grid_col = warp_id % 4; // ranges 0-3

    int l_row = (chunk_grid_row * 4) + chunk_row;
    int l_col = (chunk_grid_col * 8) + chunk_col;
    for (int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            int g_row = l_row + (i*32);
            int g_col = l_col + (j*32);
            int idx = g_row * 64 + g_col;
            s_c[idx] = idx + 1.0;
            s_a[idx] = idx + 2.0;
            s_b[idx] = idx + 1.0;
        }
    }

    // ensure all threads are done initializing the buffer
    __syncthreads();

    // Computes C += A * B using only 1 thread
    // A is column major order, the other 2 matrices are row major order
    for (int i = 0; i < 2; i++) {
        for(int j = 0; j < 2; j++) {
            int g_row = l_row + (i*32);
            int g_col = l_col + (j*32);
            int idx = g_row * 64 + g_col;
            for(int k = 0; k < 64; k++) {
                s_c[idx] += s_a[k * 64 + g_row] * s_b[k * 64 + g_col];
            }
        }
    }

    /**** Do Not Change Code Below This ****/

    // copy C out such that C is in row major order
    for (int i = id; i < 64 * 64; i += p) {
        c[i] = s_c[i];
    }
}

int main() {

    float *host_out;
    float *dev_out;

    cudaEvent_t st1, et1, st2, et2;
    cudaEventCreate(&st1);
    cudaEventCreate(&et1);
    cudaEventCreate(&st2);
    cudaEventCreate(&et2);

    float ms1, ms2;

    // create buffer on host
    host_out = (float *)malloc(64 * 64 * sizeof(float));

    // create buffer on device
    cudaError_t err = cudaMalloc(&dev_out, 64 * 64 * sizeof(float));
    if (err != cudaSuccess) {
        cout << "Dev Memory not allocated" << endl;
        exit(-1);
    }

    // record time at start
    cudaEventRecord(st2);
    
    // change number of threads here
    for(int i = 0; i < 100000; i++) {
        kernel_call<<<1, 1024>>>(dev_out);
    }
    // wait until kernel is done start timing
    cudaDeviceSynchronize();
    cudaEventRecord(et2);

    cudaEventElapsedTime(&ms2, st2, et2);
    cout << "Kernel:\t\t\t" << ms2/100000 << "ms" << endl;

    cudaMemcpy(host_out, dev_out, sizeof(float) * 64 * 64,
               cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 64; i++) {
    //     for (int j = 0; j < 64; j++) {
    //         printf("%.2f ", host_out[i*64+j]);
    //     }
    //     printf("\n");
    // }
    free(host_out);
    cudaFree(dev_out);

    return 0;
}
