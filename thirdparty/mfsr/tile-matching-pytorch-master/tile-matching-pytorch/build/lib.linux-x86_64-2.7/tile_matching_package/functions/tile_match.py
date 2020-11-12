import torch
from torch.autograd import Function
from .._ext import tile_match

import numpy as np

class tile_matching(Function):

    def __init__(self, tile_size=16, search_radius=8):
        super(tile_matching, self).__init__()
        self.tile_size = tile_size
        self.search_radius = search_radius

    def forward(self, src, dst, offset):
        assert src.is_contiguous()
        assert dst.is_contiguous()
        # shape checks
        n, c, h, w = src.size()
        num_tiles_h = h // self.tile_size
        num_tiles_w = w // self.tile_size

        assert h == num_tiles_h * self.tile_size
        assert w == num_tiles_w * self.tile_size
        _, oh, ow, oc = offset.shape
        assert oh == num_tiles_h
        assert ow == num_tiles_w
        assert oc == 2

        src_padd = src.new()
        dst_padd = dst.new()
        output = torch.zeros((n, num_tiles_h, num_tiles_w, (2*self.search_radius+1)**2), dtype=src.dtype).to(src.device)
        tile_match.tile_match_cuda_forward(src, dst, src_padd, dst_padd,
                                           offset, output, self.tile_size, self.search_radius)

        return output

    def backward(self, grad_output):
        raise NotImplementedError
