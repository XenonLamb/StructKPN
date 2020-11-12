#include <THC/THC.h>
#include "rbf_neigh_cuda_kernel.h"

extern THCState *state;

// == Forward
int rbf_neigh_cuda_forward(
        THCudaTensor *r,
        THCudaTensor *c,
        THCudaTensor *p_neigh,
        THCudaTensor *a_neigh,
        THCudaTensor *dx_neigh,
        THCudaTensor *dy_neigh,
        int radius,
        int k,
        int h,
        int w)
{
    int batch_size = r->size[0];
    int length = c->size[1];

    // Inputs
    float *r_data = THCudaTensor_data(state, r);
    float *c_data = THCudaTensor_data(state, c);

    // Output
    THCudaTensor_resize4d(state, p_neigh, batch_size, h, w, k); 
    THCudaTensor_zero(state, p_neigh);
    float *p_neigh_data = THCudaTensor_data(state, p_neigh);

    THCudaTensor_resize4d(state, a_neigh, batch_size, h, w, k); 
    THCudaTensor_zero(state, a_neigh);
    float *a_neigh_data = THCudaTensor_data(state, a_neigh);

    THCudaTensor_resize4d(state, dx_neigh, batch_size, h, w, k); 
    THCudaTensor_zero(state, dx_neigh);
    float *dx_neigh_data = THCudaTensor_data(state, dx_neigh);

    THCudaTensor_resize4d(state, dy_neigh, batch_size, h, w, k); 
    THCudaTensor_zero(state, dy_neigh);
    float *dy_neigh_data = THCudaTensor_data(state, dy_neigh);

    cudaStream_t stream = THCState_getCurrentStream(state);

    rbf_neigh_on_gpu(
            r_data, c_data,
            p_neigh_data, a_neigh_data, dx_neigh_data, dy_neigh_data,
            radius, k, batch_size, length, h, w, stream);

    return 1;

}

