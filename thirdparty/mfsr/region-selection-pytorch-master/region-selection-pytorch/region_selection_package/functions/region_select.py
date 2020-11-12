import torch
from torch.autograd import Function
from .._ext import region_select
import math

import numpy as np

class region_selection(Function):

    def __init__(self, h, w):
        super(region_selection, self).__init__()
        self.h = h
        self.w = w

    def forward(self, x, indices):
        assert x.is_contiguous()
        h, w = self.h, self.w
        # shape checks
        xn, xh, xw, mh, mw = x.size()
        idn, idh, idw, ch = indices.size()
        assert idn == xn and idh == xh and idw == xw and ch == 2

        x = x.view(xn, xh, xw, -1)
        output = torch.zeros((xn, xh, xw, h*w), dtype=x.dtype).to(x.device)
        region_select.region_select_cuda_forward(x, indices, output, mw, mw, h, w)
        output = output.view(xn, xh, xw, h, w)

        return output

    def backward(self, grad_output):
        raise NotImplementedError
