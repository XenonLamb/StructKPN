#include <THC/THC.h>
#include "region_select_cuda_kernel.h"

extern THCState *state;

// == Forward
int region_select_cuda_forward(
        THCudaTensor *x,
        THCudaTensor *indices,
        THCudaTensor *output,
        int map_height,
        int map_width,
        int region_height,
        int region_width)
{
    int batch_size = x->size[0];
    int in_height = x->size[1];
    int in_width = x->size[2];
    int region_size = region_height * region_width;

    // Inputs
    float *x_data = THCudaTensor_data(state, x);
    float *idx_data = THCudaTensor_data(state, indices);

    // Output
    THCudaTensor_resize4d(state, output, batch_size, in_height, in_width, region_size); 
    THCudaTensor_zero(state, output);
    float *output_data = THCudaTensor_data(state, output);

    cudaStream_t stream = THCState_getCurrentStream(state);

    select_on_gpu(x_data, idx_data, output_data, 
            batch_size, in_height, in_width, 
            map_height, map_width,
            region_height, region_width, 
            stream);

    return 1;

}

