import os
import torch
from torch.utils.ffi import create_extension


sources = ['rbf_neighbours_package/src/rbf_neigh.c']
headers = ['rbf_neighbours_package/src/rbf_neigh.h']

defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['rbf_neighbours_package/src/rbf_neigh_cuda.c']
    headers += ['rbf_neighbours_package/src/rbf_neigh_cuda.h']

    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = ['rbf_neighbours_package/src/rbf_neigh_cuda_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    'rbf_neighbours_package._ext.rbf_neigh',
    package=True,
    headers=headers,
    sources=sources,
    define_macros=defines,
    relative_to=__file__,
    with_cuda=with_cuda,
    extra_objects=extra_objects,
)

if __name__ == '__main__':
    ffi.build()
