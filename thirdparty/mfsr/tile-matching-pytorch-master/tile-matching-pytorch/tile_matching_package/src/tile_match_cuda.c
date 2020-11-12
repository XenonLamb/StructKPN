#include <THC/THC.h>
#include "tile_match_cuda_kernel.h"

extern THCState *state;

// == Forward
int tile_match_cuda_forward(
        THCudaTensor *src,
        THCudaTensor *dst,
        THCudaTensor *src_p,
        THCudaTensor *dst_p,
        THCudaTensor *offset,
        THCudaTensor *output,
        int tile_size,
        int search_radius)
{
    int batch_size = src->size[0];
    int in_channels = src->size[1];
    int in_height = src->size[2];
    int in_width = src->size[3];
    int num_tiles_h = in_height / tile_size;
    int num_tiles_w = in_width / tile_size;

    long in_imsize = in_height * in_width;
    int padd_height = in_height + 2 * search_radius;
    int padd_width = in_width + 2 * search_radius;
    int search_size = 2 * search_radius + 1;
    int out_channels = search_size * search_size;

    // Inputs
    float *src_data = THCudaTensor_data(state, src);
    float *dst_data = THCudaTensor_data(state, dst);
    float *offset_data = THCudaTensor_data(state, offset);

    // Output
    THCudaTensor_resize4d(state, output, batch_size, num_tiles_h, num_tiles_w, out_channels); 
    THCudaTensor_zero(state, output);
    float *output_data = THCudaTensor_data(state, output);

    // Padded input 
    THCudaTensor_resize4d(state, src_p, batch_size, in_height, in_width, in_channels);
    THCudaTensor_resize4d(state, dst_p, batch_size, padd_height, padd_width, in_channels);
    THCudaTensor_zero(state, src_p);
    THCudaTensor_zero(state, dst_p);
    float *src_p_data = THCudaTensor_data(state, src_p);
    float *dst_p_data = THCudaTensor_data(state, dst_p);

    cudaStream_t stream = THCState_getCurrentStream(state);

    // pad input and permute from NCHW to NHWC
    long padd_imsize = padd_height * padd_width;
    padd_on_gpu(src_data, src_p_data, batch_size, in_channels,
            in_height, in_width, in_imsize, 0, in_imsize, stream);
    padd_on_gpu(dst_data, dst_p_data, batch_size, in_channels, 
            in_height, in_width, in_imsize, search_radius, padd_imsize, stream);

    match_on_gpu(src_p_data, dst_p_data, offset_data, output_data, 
            tile_size, batch_size, in_height, in_width, in_channels, 
            num_tiles_h, num_tiles_w, search_radius, search_size, 
            padd_height, padd_width, stream);

    THCudaTensor_free(state, src_p);
    THCudaTensor_free(state, dst_p);

    return 1;

}

