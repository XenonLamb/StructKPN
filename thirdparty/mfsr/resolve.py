from __future__ import print_function
import os, sys
# import cv2

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import utils

from rbf_neighbours_package.modules.rbf_neigh import RBFNeighbours
from .resample2d_package.modules.resample2d import Resample2d

'''
Local structure analysis. It analyses the directions and strength of local
gradients for each pixel of the given image, by computing the eigen vectors
and values of local covariance matrix of pixel graidents. See the paper
"Handheld Multi-frame Super-resolution, SIGGRAPH 2019" for details.
'''
class LocalStructureAnalyse(nn.Module):
    def __init__(self, cfg, scaling_factor):
        super(LocalStructureAnalyse, self).__init__()
        self.cfg = cfg
        self.scaling_factor = scaling_factor
        self.weight = torch.Tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).view((1, 1, 3, 3)) / 16.

    '''
    Input: NCHW image tensor, where C must equal 1
    Output: eigen values and vectors of shape NHW
    '''
    def forward(self, x):
        n, c, h, w = x.size()

        sh, sw = int(h * self.scaling_factor), int(w * self.scaling_factor)
        x = F.upsample(x, size=(sh, sw), mode='bilinear', align_corners=True)

        gx = F.pad(x[:, :, :, :-1], (1, 0, 0, 0), mode='reflect') - x
        gy = F.pad(x[:, :, :-1, :], (0, 0, 1, 0), mode='reflect') - x

        weight = self.weight.to(x.device)
        Axx, Axy, Ayy = 0., 0., 0.
        for _ in range(c):
            gx_, gy_ = gx[:, _:_+1, :, :], gy[:, _:_+1, :, :]
            Axx += F.conv2d(gx_ * gx_, weight, padding=1)
            Axy += F.conv2d(gx_ * gy_, weight, padding=1)
            Ayy += F.conv2d(gy_ * gy_, weight, padding=1)
        Axx /= c
        Axy /= c
        Ayy /= c

        Axx = Axx.permute(0, 2, 3, 1)
        Axy = Axy.permute(0, 2, 3, 1)
        Ayy = Ayy.permute(0, 2, 3, 1)

        A = torch.cat([Axx, Axy, Axy, Ayy], dim=3)
        A = A.view(n, sh, sw, 2, 2)

        lambda1, lambda2, e11, e12, e21, e22 = utils.eig2x2(A)

        return lambda1, lambda2, e11, e12, e21, e22

    def backward(self, x):
        print('%s: this module does not support back propagation' %(self.__name__), file=sys.stderr)
        raise

'''
RBF kernel construction
'''
class RBFKernel(nn.Module):
    def __init__(self, cfg):
        super(RBFKernel, self).__init__()
        self.cfg = cfg

    def forward(self, lamb1, lamb2, e11, e12, e21, e22):

        # Unit-scaling the eigen vectors
        e1_len = torch.sqrt(e11**2 + e12**2)
        e2_len = torch.sqrt(e21**2 + e22**2)
        e11 = torch.div(e11, e1_len)
        e12 = torch.div(e12, e1_len)
        e21 = torch.div(e21, e2_len)
        e22 = torch.div(e22, e2_len)

        # Kernel variance
        A = 1.0 + torch.sqrt(lamb1 / lamb2)
        k1_ = self.cfg.K_DETAIL * self.cfg.K_STRETCH * A
        k2_ = self.cfg.K_DETAIL / (self.cfg.K_SHRINK * A)
        D = torch.clamp(1 - torch.sqrt(lamb2) / self.cfg.D_TR + self.cfg.D_TH, 0, 1)

        k1 = (1 - D) * k1_ + D * (self.cfg.K_DETAIL * self.cfg.K_DENOISE)**2
        k2 = (1 - D) * k2_ + D * (self.cfg.K_DETAIL * self.cfg.K_DENOISE)**2

        # Compute RBF kernel
        o11 = k1 * e11 * e11 + k2 * e21 * e21
        o12 = k1 * e11 * e12 + k2 * e21 * e22
        o21 = k1 * e11 * e12 + k2 * e21 * e22
        o22 = k1 * e12 * e12 + k2 * e22 * e22

        # Compute inverse RBF kernel
        det = o11 * o22 - o12 * o21
        o11_inv = o22 / det
        o12_inv = -o12 / det
        o21_inv = -o21 / det
        o22_inv = o11 / det

        return o11_inv, o12_inv, o21_inv, o22_inv

'''
Motion robustness
'''
class MotionRobustness(nn.Module):
    def __init__(self, cfg):
        super(MotionRobustness, self).__init__()
        self.s, self.t = cfg.SCALE, cfg.THRESH
        self.win_radius = cfg.WIN_RADIUS
        self.win_size = cfg.WIN_RADIUS * 2 + 1
        self.neigh_size = self.win_size**2
        self.morph_radius = cfg.MORPH_RADIUS
        self.morph_size = 2 * cfg.MORPH_RADIUS + 1
        self.box_filter = (torch.ones(self.neigh_size) / float(self.neigh_size)) \
            .view((1, 1, self.win_size, self.win_size))
        self.warp = Resample2d()

    def forward(self, x_ref, x_alt, offset):
        n, c, h, w = x_ref.size()
        _, ca, _, _ = x_alt.size()
        assert c == 1
        assert ca == 1

        offset_t = torch.stack([offset[:, 1, :, :], offset[:, 0, :, :]], dim=1)

        # warp reference to alternate coordinates
        x_ref_warp = self.warp(x_ref, offset_t)

        # compute gradients
        bf = self.box_filter.to(x_ref.device)
        mx = F.conv2d(x_alt, bf, padding=self.win_radius)
        vx = F.conv2d(x_alt**2, bf, padding=self.win_radius) - mx**2
        mxw = F.conv2d(x_ref_warp, bf, padding=self.win_radius)
        dx = torch.abs(mx - mxw)
        p2d = tuple([self.morph_radius]*4)
        rx = torch.clamp(self.s*torch.exp(-torch.pow(dx, 2) / (vx + 1e-6)) - self.t, 0., self.s-self.t)
        rx_padd = F.pad(rx, p2d, mode='constant', value=self.s-self.t)
        q = -F.max_pool2d(-rx_padd, kernel_size=self.morph_size, stride=1)
        q = rx

        return q[:, 0, :, :]

    def backward(self, x):
        print('%s: this module does not support back propagation' %(self.__name__), file=sys.stderr)
        raise

'''
Super-resolution. We find for each output pixel the nearest neighbors in
the alternate images, and aggregate their values based on RBF weights
as well as the predicted pixel qualities. Aggregation is implemented
with forward weighted rendering.
'''
class SuperResolve(nn.Module):
    def __init__(self, scaling_factor, cfg):
        super(SuperResolve, self).__init__()
        self.cfg = cfg
        self.scaling_factor = scaling_factor
        self.unsharp_mask = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]).view((1, 1, 3, 3))

    def forward(self, alt_list, o11, o12, o21, o22, offset_list, q_list):
        n, c, h, w = alt_list[0].size()

        # output indices in the super-resolved image
        device = alt_list[0].device
        sh, sw = int(self.scaling_factor * h), int(self.scaling_factor * w)
        r_idx_ = np.matmul(np.arange(0, self.scaling_factor*h, self.scaling_factor).reshape(-1, 1), np.ones((1, w)))
        c_idx_ = np.matmul(np.ones((h, 1)), np.arange(0, self.scaling_factor*w, self.scaling_factor).reshape(1, -1))
        r_idx_, c_idx_ = r_idx_.flatten(), c_idx_.flatten()
        r_idx_ = torch.FloatTensor(r_idx_).unsqueeze(0).expand(n, -1).to(device)
        c_idx_ = torch.FloatTensor(c_idx_).unsqueeze(0).expand(n, -1).to(device)
        r_idx = r_idx_.long()
        c_idx = c_idx_.long()

        x_out, weights = 0., 0.
        radius = int(self.scaling_factor * self.cfg.RBF_RADIUS)
        find_rbf_neighbours = RBFNeighbours(radius, self.cfg.MAX_NEIGHBOURS, sh, sw)
        unsharp_mask = self.unsharp_mask.to(alt_list[0].device)
        for alt, offset, q in zip(alt_list, offset_list, q_list):

            # shifted indices
            r_idx_sh = r_idx_ + offset[:, 0, :, :].view(n, -1) * self.scaling_factor
            c_idx_sh = c_idx_ + offset[:, 1, :, :].view(n, -1) * self.scaling_factor

            p_neigh, a_neigh, dx_neigh, dy_neigh = find_rbf_neighbours(r_idx_sh, c_idx_sh)
            p_neigh = p_neigh.view(n, -1).long()
            r_neigh = utils.batched_index_select(r_idx_, 1, p_neigh).view(n, sh, sw, -1) / self.scaling_factor
            c_neigh = utils.batched_index_select(c_idx_, 1, p_neigh).view(n, sh, sw, -1) / self.scaling_factor
            r_neigh, c_neigh = torch.round(r_neigh).long(), torch.round(c_neigh).long()
            dx_neigh /= self.scaling_factor
            dy_neigh /= self.scaling_factor

            del p_neigh, r_idx_sh, c_idx_sh

            # weighted aggregation
            rbf_dist = dx_neigh * dx_neigh * o11.unsqueeze(3) + dx_neigh * dy_neigh * o12.unsqueeze(3) + \
                dy_neigh * dx_neigh * o21.unsqueeze(3) + dy_neigh * dy_neigh * o22.unsqueeze(3)

            rbf_dist = torch.clamp(rbf_dist, 0., 20.)
            rbf_weights = torch.exp(-0.5 * rbf_dist)

            del rbf_dist

            q_weights = utils.batched_index_select_2d(q, r_neigh, c_neigh)

            del dx_neigh, dy_neigh

            weights_ = torch.sum(rbf_weights * q_weights * a_neigh, dim=3)
            x_out_ = []
            for ch in range(c):
                alt_ch = alt[:, ch, :, :]
                alt_neigh = utils.batched_index_select_2d(alt_ch, r_neigh, c_neigh)
                x_out_.append(torch.sum(alt_neigh * rbf_weights * q_weights * a_neigh, dim=3))
            x_out_ = torch.stack(x_out_, dim=1)

            del rbf_weights, q_weights, r_neigh, c_neigh

            weights += weights_
            x_out += x_out_

            del weights_, x_out_

        for _ in range(c):
            x_out[:, _, :, :] = x_out[:, _, :, :] / weights

        return x_out

