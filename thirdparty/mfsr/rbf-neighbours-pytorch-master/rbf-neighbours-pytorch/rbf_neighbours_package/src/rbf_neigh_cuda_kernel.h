#ifndef _RBF_NEIGH_CUDA_KERNEL
#define _RBF_NEIGH_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

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
        cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
