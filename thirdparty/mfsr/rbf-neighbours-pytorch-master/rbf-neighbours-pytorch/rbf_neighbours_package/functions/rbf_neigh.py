import torch
from torch.autograd import Function
from .._ext import rbf_neigh
import math

import numpy as np

class rbf_neighbours(Function):

    def __init__(self, radius, k, h, w):
        super(rbf_neighbours, self).__init__()
        self.radius = radius
        self.k = k
        self.h = h
        self.w = w

    def forward(self, r, c):
        assert len(r.shape) == len(c.shape) == 2
        rn, rl = r.shape
        cn, cl = c.shape
        assert rn == cn
        assert rl == cl

        radius, h, w, k = self.radius, self.h, self.w, self.k
        p_neigh = torch.zeros((rn, h, w, k), dtype=r.dtype).to(r.device)
        a_neigh = torch.zeros((rn, h, w, k), dtype=r.dtype).to(r.device)
        dx_neigh = torch.zeros((rn, h, w, k), dtype=r.dtype).to(r.device)
        dy_neigh = torch.zeros((rn, h, w, k), dtype=r.dtype).to(r.device)
        rbf_neigh.rbf_neigh_cuda_forward(r, c, p_neigh, a_neigh, dx_neigh, dy_neigh,
                                         radius, k, h, w)
        return p_neigh, a_neigh, dx_neigh, dy_neigh

    def backward(self, grad_output):
        raise NotImplementedError
