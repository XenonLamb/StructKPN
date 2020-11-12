int region_select_cuda_forward(
        THCudaTensor *x,
        THCudaTensor *indices,
        THCudaTensor *output,
        int map_height,
        int map_width,
        int region_height,
        int region_width);
