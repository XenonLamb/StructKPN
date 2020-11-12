#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "rbf_neigh_cuda_kernel.h"

#define CUDA_NUM_THREADS 1024
#define CUDA_KERNEL_LOOP(i, n) for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#define GET_BLOCKS(n, t) (n+t-1) / t

// Kernel for neighbour collection 
__global__ void rbf_neigh_kernel(
        long nthreads,
        int offset_r,
        int offset_c,
        const float *r,
        const float *c,
        float *p_neigh,
        float *a_neigh,
        float *dx_neigh,
        float *dy_neigh,
        int radius,
        int k,
        int batch_size,
        int length,
        int h,
        int w)
{
    CUDA_KERNEL_LOOP(index, nthreads) {
        int idx_p = index % length; // the index of points 
        int item = index / length; // batch size

        // input coordinates
        float r_in = r[item * length + idx_p];
        float c_in = c[item * length + idx_p];

        // round coordinates
        float r_in_int = roundf(r_in);
        float c_in_int = roundf(c_in);

        // offset position
        int r_off_int = r_in_int + offset_r;
        int c_off_int = c_in_int + offset_c;

        float dy = r_off_int * 1.0 - r_in;
        float dx = c_off_int * 1.0 - c_in;
        float dist = dy * dy + dx * dx;

        if (r_off_int >= 0 && r_off_int < h && c_off_int >= 0 && c_off_int < w) {
            if (fabs(dx) < radius + 1e-6 && fabs(dy) < radius + 1e-6) {
                // loop current neighbours and replace the farest one
                float max_dist = -1.0;
                int max_ch = -1;
                bool updated = false;
                for(int ch = 0; ch < k; ch++) {
                    // output index
                    int idx_out = ((item * h + r_off_int) * w + c_off_int) * k + ch;
                    if(a_neigh[idx_out] < 1) { // fill the empty location
                        a_neigh[idx_out] = 1;
                        p_neigh[idx_out] = idx_p;
                        dx_neigh[idx_out] = dx;
                        dy_neigh[idx_out] = dy;
                        updated = true;
                        break;
                    } else {
                        float tmp_dist = dx_neigh[idx_out] * dx_neigh[idx_out] + dy_neigh[idx_out] * dy_neigh[idx_out];
                        if (a_neigh[idx_out] > 0 && tmp_dist > max_dist) {
                            max_dist = tmp_dist;
                            max_ch = ch;
                        }
                    }
                }
                if(!updated && dist < max_dist) { // replace the farest one
                    int idx_out = ((item * h + r_off_int) * w + c_off_int) * k + max_ch;
                    a_neigh[idx_out] = 1;
                    p_neigh[idx_out] = idx_p;
                    dx_neigh[idx_out] = dx;
                    dy_neigh[idx_out] = dy;
                }
            }
        }
    }

    __syncthreads();
}

void rbf_neigh_on_gpu(
        const float *r,
        const float *c,
        float *p_neigh,
        float *a_neigh,
        float *dx_neigh,
        float *dy_neigh,
        int radius,
        int k,
        int batch_size, 
        int length,
        int h,
        int w,
        cudaStream_t stream)
{
    int output_thread_count = batch_size * length;

    cudaError_t err;

    for(int offset_r = -radius; offset_r <= radius; offset_r++) { 
        for(int offset_c = -radius; offset_c <= radius; offset_c++) {
            rbf_neigh_kernel<<<GET_BLOCKS(output_thread_count, CUDA_NUM_THREADS), CUDA_NUM_THREADS, 0, stream>>>(
                    output_thread_count, offset_r, offset_c,
                    r, c, p_neigh, a_neigh, dx_neigh, dy_neigh,
                    radius, k, batch_size, length, h, w);

            err = cudaGetLastError();
            if(cudaSuccess != err)
            {
                fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
                exit(-1);
            }
        }
    }
}

