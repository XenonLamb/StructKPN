#ifndef _TILE_MATCH_CUDA_KERNEL
#define _TILE_MATCH_CUDA_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

void padd_on_gpu(
        const float *in,
        float *out,
        int num,
        int channels,
        int height,
        int width,
        long widthheight,
        int padd,
        long pwidthheight,
        cudaStream_t stream
        );

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
        cudaStream_t stream
        );

#ifdef __cplusplus
}
#endif

#endif
