#ifndef _REGION_SELECT_CUDA_KERNEL
#define _REGION_SELECT_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

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
        cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
