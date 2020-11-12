int tile_match_cuda_forward(
        THCudaTensor *src,
        THCudaTensor *dst,
        THCudaTensor *src_p,
        THCudaTensor *dst_p,
        THCudaTensor *offset,
        THCudaTensor *output,
        int tile_size,
        int search_radius);

