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
        int w);