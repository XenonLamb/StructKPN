import os
import torch
from torch.utils.ffi import create_extension


sources = ['region_selection_package/src/region_select.c']
headers = ['region_selection_package/src/region_select.h']

defines = []
with_cuda = False

if torch.cuda.is_available():
    print('Including CUDA code.')
    sources += ['region_selection_package/src/region_select_cuda.c']
    headers += ['region_selection_package/src/region_select_cuda.h']

    defines += [('WITH_CUDA', None)]
    with_cuda = True

this_file = os.path.dirname(os.path.realpath(__file__))
extra_objects = ['region_selection_package/src/region_select_cuda_kernel.cu.o']
extra_objects = [os.path.join(this_file, fname) for fname in extra_objects]

ffi = create_extension(
    'region_selection_package._ext.region_select',
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
