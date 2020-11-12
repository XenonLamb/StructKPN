#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "region_select_cuda_kernel.h"

// Kernel for selection 
__global__ void select_kernel(
        const float *x,
        const float *indices,
        float *output,
        int *bdry_error,
        int batch_size,
        int in_height,
        int in_width,
        int map_height,
        int map_width,
        int region_height,
        int region_width)
{
  // Batch
  int item = blockIdx.x;
  // Row 
  int xr = blockIdx.y;
  // Column 
  int xc = blockIdx.z;

  // Region index
  int rr = threadIdx.x;
  int rc = threadIdx.y;

  // Anchor position
  int ar = indices[((item * in_height + xr) * in_width + xc) * 2];
  int ac = indices[((item * in_height + xr) * in_width + xc) * 2 + 1];

  // Shifted position
  int sr = ar + rr - region_height / 2;
  int sc = ac + rc - region_width / 2;

  // Boundary error
  int error = (sr < 0 || sr >= map_height || sc < 0 || sc >= map_width);

  // input index
  int idx_in = (((item * in_height + xr) * in_width + xc) * map_height + sr) * map_width + sc;

  // output index
  int idx_out = (((item * in_height + xr) * in_width + xc) * region_height + rr) * region_width + rc;

  output[idx_out] = x[idx_in];

  __syncthreads();

  if (error > 0)
      *bdry_error = 1;
    
  __syncthreads();
}

void select_on_gpu(
        const float *x, 
        const float *indices, 
        float *output, 
        int batch_size, 
        int in_height,
        int in_width,
        int map_height, 
        int map_width,
        int region_height,
        int region_width, 
        cudaStream_t stream)
{
    dim3 threadsPerBlock(region_height, region_width, 1);
    dim3 totalBlocksMatch(batch_size, in_height, in_width);
    cudaError_t err;

	int bdry_error = 0;
    select_kernel<<<totalBlocksMatch, threadsPerBlock, 0, stream>>>(
            x, indices, output, &bdry_error,
            batch_size, in_height, in_width,
            map_height, map_width,
            region_height, region_width);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    if(bdry_error > 0) {
        fprintf(stderr, "Index exceeds image boundary\n");
        exit(-1);
    }
}

