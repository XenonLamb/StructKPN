#!/usr/bin/env bash
export CXXFLAGS="-std=c++11"
export CFLAGS="-std=c99"

NVCC_PATH=/usr/local/cuda/bin/nvcc

cd region-selection-pytorch/region_selection_package/src
echo "Compiling kernels of region selection layer by nvcc..."

# TODO (JEB): Check which arches we need
$NVCC_PATH -c -o region_select_cuda_kernel.cu.o region_select_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52 

cd ../../
python setup.py build install
