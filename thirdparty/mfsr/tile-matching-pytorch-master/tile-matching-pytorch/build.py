import os
import torch
from torch.utils.ffi import create_extension


sources = ['tile_matching_package/src/tile_match.c']
headers = ['tile_matching_package/src/tile_match.h']

defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['tile_matching_package/src/tile_match_cuda.c']
    headers += ['tile_matching_package/src/tile_match_cuda.h']

    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = ['tile_matching_package/src/tile_match_cuda_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    'tile_matching_package._ext.tile_match',
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
