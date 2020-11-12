#include <vector>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "tile_match_cuda_kernel.h"

// Kernel for padding
__global__ void padd_kernel(
        const float *in, 
        float *out, 
        int num, 
        int channels, 
        int height, 
        int width, 
        long widthheight, 
        int padding, 
        long pwidthheight)
{
    int xy = blockIdx.x*blockDim.x + threadIdx.x;
    if(xy>=widthheight)
        return;

    int ch = blockIdx.y;
    int n  = blockIdx.z;


    float value=in[(n*channels+ch)*widthheight+xy];

    __syncthreads();

    int xpad  = (xy % width + padding);
    int ypad  = (xy / width + padding);
    int xypad = ypad * (width+2*padding) + xpad;

    out[(n*pwidthheight+xypad)*channels + ch] = value;
}

void padd_on_gpu(
        const float *in,
        float *out, 
        int num, 
        int channels, 
        int height, 
        int width, 
        long widthheight, 
        int padding, 
        long pwidthheight, 
        cudaStream_t stream)
{
    int threads_per_block = 16;
    dim3 totalBlocksRearr((widthheight-1)/threads_per_block+1, channels, num);

    cudaError_t err;

    padd_kernel<<<totalBlocksRearr, threads_per_block, 0, stream>>>
        (in, out, num, channels, height, width, widthheight, padding, pwidthheight);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

// Kernel for matching
__global__ void match_kernel(
        const float *in_src,
        const float *in_dst,
        const float *offset,
        float *output,
        int *offset_error,
        int tile_size,
        int in_height,
        int in_width,
        int in_channels,
        int num_tiles_h,
        int num_tiles_w,
        int search_radius,
        int search_size,
        int padd_height,
        int padd_width)
{
  // The batch
  int item = blockIdx.z;
  // Upper left position of the tile in the src image
  int x0 = blockIdx.x * tile_size;
  int y0 = blockIdx.y * tile_size;

  __syncthreads();

  // Offset for search
  int idx_off_y = ((item * num_tiles_h + blockIdx.y) * num_tiles_w + blockIdx.x) * 2;
  int idx_off_x = idx_off_y + 1;
  int xs = threadIdx.x - search_radius;
  int ys = threadIdx.y - search_radius;

  // upper left index of the tile in the dst image
  int x1 = x0 + search_radius + xs + offset[idx_off_x];
  int y1 = y0 + search_radius + ys + offset[idx_off_y];

  int error = (y1 < 0 || y1 > padd_height - tile_size || x1 < 0 || x1 > padd_width - tile_size);

  // Compute L2 distance
  float sum = 0;
  for(int j = 0; j < tile_size; j++) { // height
      for(int i = 0; i < tile_size; i++) { // width
          for(int ch = 0; ch < in_channels; ch++) { // channels
              // index of src image
              int idx_src = ((item * in_height + y0 + j) * in_width + x0 + i) * in_channels + ch;
              int idx_dst = ((item * padd_height + y1 + j) * padd_width + x1 + i) * in_channels + ch;
              sum += (in_src[idx_src] - in_dst[idx_dst]) * (in_src[idx_src] - in_dst[idx_dst]);
          }
      }
  }
  sum /= (tile_size * tile_size * in_channels);

  // index of the output tensor
  int xo = search_radius + xs;
  int yo = search_radius + ys;
  int xyo = yo * search_size + xo;
  int idx_out = ((item * num_tiles_h + blockIdx.y) * num_tiles_w + blockIdx.x) * search_size * search_size + xyo;
  output[idx_out] = sum;

  __syncthreads();

  if (error > 0)
      *offset_error = 1;
    
  __syncthreads();
}

void match_on_gpu(
        const float *in_src, 
        const float *in_dst, 
        const float *offset,
        float *output, 
        int tile_size,
        int batch_size, 
        int in_height,
        int in_width,
        int in_channels, 
        int num_tiles_h,
        int num_tiles_w,
        int search_radius, 
        int search_size, 
        int padd_height,
        int padd_width, 
        cudaStream_t stream)
{
    dim3 threadsPerBlock(search_size, search_size, 1);
    dim3 totalBlocksMatch(num_tiles_w, num_tiles_h, batch_size);
    cudaError_t err;

	int offset_error = 0;
    match_kernel<<<totalBlocksMatch, threadsPerBlock, 0, stream>>>(
            in_src, in_dst, offset, output, &offset_error,
            tile_size, in_height, in_width, in_channels,
            num_tiles_h, num_tiles_w,
            search_radius, search_size, padd_height, padd_width
            );

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf(stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    if(offset_error > 0) {
        fprintf(stderr, "Invalid offset encountered\n");
        exit(-1);
    }
}

