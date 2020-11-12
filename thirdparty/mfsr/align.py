from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from tile_matching_package.modules.tile_match import TileMatching
from region_selection_package.modules.region_select import RegionSelection
from . import utils

'''
Gaussian pyramid construction
'''
class GaussianPyramid(nn.Module):
    def __init__(self, num_levels, ds_factor, radius=2, sigma=1.0):
        super(GaussianPyramid, self).__init__()
        window_size = 2 * radius + 1
        self.num_levels = num_levels
        self.ds_factor = ds_factor
        self.window_size = window_size
        self.sigma = sigma

    '''
    Input: a NCHW pytorch image tensor
    Output: a list of tensors representing the gaussian pyramid
    '''
    def forward(self, x):
        _, c, h, w = x.size()
        real_size = min(self.window_size, h, w)
        window = utils.create_gaussian_kernel(channel=c,
                                              window_size=real_size,
                                              sigma=self.sigma)
        window = window.to(x.device)
        pyram = [x]
        for i in range(1, self.num_levels):
            x = F.conv2d(x, window, padding=(self.window_size-1)//2, groups=c)
            x = F.avg_pool2d(x, kernel_size=self.ds_factor, stride=self.ds_factor)
            pyram.append(x)
        return pyram

'''
HDR+ tile alignment algorithm. We use an offset matrix of shape (num_tile_x,
num_tile_y, 2) to represent alignment, thus offset[tx,ty]=[ox, oy] are
the offset vector of the (tx, ty)th tile.
Details referred to: Burst Photography for High Dynamic Range and Low-Light
Imaging on Mobile Cameras, SIGGRAPH 2016.
'''
class HDRPlusAlign(nn.Module):
    def __init__(self, cfg):
        super(HDRPlusAlign, self).__init__()
        self.cfg = cfg

        # set filters for subpixel fitting (see the supplemenatry material of
        # HDR+ paper)
        self.f_a11 = 1.0 / 4 * torch.FloatTensor([[ 1, -2,  1],
                                                  [ 2, -4,  2],
                                                  [ 1, -2,  1]])
        self.f_a22 = 1.0 / 4 * torch.FloatTensor([[ 1,  2,  1],
                                                  [-2, -4, -2],
                                                  [ 1,  2,  1]])
        self.f_a12 = 1.0 / 4 * torch.FloatTensor([[ 1,  0, -1],
                                                  [ 0,  0,  0],
                                                  [-1,  0,  1]])
        self.f_a21 = self.f_a12.clone()
        self.f_b1 = 1.0 / 8 * torch.FloatTensor([[-1,  0,  1],
                                                 [-2,  0,  2],
                                                 [-1,  0,  1]])
        self.f_b2 = 1.0 / 8 * torch.FloatTensor([[-1, -2, -1],
                                                 [ 0,  0,  0],
                                                 [ 1,  2,  1]])
        self.f_c = 1.0 / 16 * torch.FloatTensor([[-1,  2, -1],
                                                 [ 2, 12,  2],
                                                 [-1,  2, -1]])

        # a spatial distance map for matching result smoothing
        spatial_weight = self.cfg.SPATIAL_WEIGHT
        search_radius = self.cfg.SEARCH_RADIUS
        padd = 1 # for subpixel fitting
        y_idx, x_idx = utils.get_matrix_indices(2*(search_radius+padd)+1, 2*(search_radius+padd)+1)
        y_idx -= (search_radius + padd)
        x_idx -= (search_radius + padd)
        y_idx = y_idx * 1.0 / (2 * (search_radius + padd) + 1)
        x_idx = x_idx * 1.0 / (2 * (search_radius + padd) + 1)
        sdist_map = (y_idx ** 2 + x_idx ** 2).reshape(2*(search_radius+padd)+1, 2*(search_radius+padd)+1)
        self.sdist_map = torch.Tensor(sdist_map).float()

        # Gaussian pyramid layer
        self.build_pyramid = GaussianPyramid(num_levels=self.cfg.NUM_LEVELS,
                                             ds_factor=self.cfg.DS_FACTOR,
                                             radius=self.cfg.GAUSS_RADIUS,
                                             sigma=self.cfg.GAUSS_SIGMA)

        # tile matching layer
        self.tile_match = TileMatching(tile_size=self.cfg.TILE_SIZE,
                                       search_radius=search_radius+padd)
        # region selection layer
        self.region_select = RegionSelection(3, 3)

    '''
    Get tile index in parent level given that in current level
    '''
    def idx_pt(self, t):
        img_idx = self.idx_im(t)
        img_idx_par = img_idx / self.cfg.DS_FACTOR
        res = img_idx_par / self.cfg.TILE_SIZE
        return res.astype(np.int32) if type(res) is np.ndarray else int(res)

    '''
    Get image index of a given tile
    '''
    def idx_im(self, t):
        res = self.cfg.TILE_SIZE * t
        return res.astype(np.int32) if type(res) is np.ndarray else int(res)

    '''
    Get the number of tiles given image size
    '''
    def image_size_to_tile_num(self, ims):
        return int(np.ceil(ims / self.cfg.TILE_SIZE))

    '''
    Inherit tile offsets from previous level
    '''
    def inherit_offset(self, level, prev_offset):
        tile_size = self.cfg.TILE_SIZE
        # get offset matrix shape
        n, _, _, _ = prev_offset.size()
        img_size_h, img_size_w = self.pyram_sizes[level]
        num_tile_h = self.image_size_to_tile_num(img_size_h)
        num_tile_w = self.image_size_to_tile_num(img_size_w)
        offset = torch.zeros(n, num_tile_h, num_tile_w, 2).to(prev_offset.device)

        # inherit offset
        ty_idx, tx_idx = utils.get_matrix_indices(num_tile_h, num_tile_w)
        ty_idx_prev = self.idx_pt(ty_idx)
        tx_idx_prev = self.idx_pt(tx_idx)
        offset[:, ty_idx, tx_idx, :] = self.cfg.DS_FACTOR * prev_offset[:, ty_idx_prev, tx_idx_prev, :]

        # clip invalid offsets (not a justified case, should rarely happen)
        iy_idx = np.expand_dims(self.idx_im(ty_idx), 0).repeat(n, axis=0) # [n, len(ty_idx)]
        ix_idx = np.expand_dims(self.idx_im(tx_idx), 0).repeat(n, axis=0)
        iy_idx_ = torch.from_numpy(iy_idx).float().to(prev_offset.device)
        ix_idx_ = torch.from_numpy(ix_idx).float().to(prev_offset.device)
        dst_ty_idx = offset[:, ty_idx, tx_idx, 0] + iy_idx_
        dst_tx_idx = offset[:, ty_idx, tx_idx, 1] + ix_idx_
        dst_ty_idx = torch.clamp(dst_ty_idx, 0, img_size_h-tile_size)
        dst_tx_idx = torch.clamp(dst_tx_idx, 0, img_size_w-tile_size)
        offset[:, ty_idx, tx_idx, 0] = dst_ty_idx - iy_idx_
        offset[:, ty_idx, tx_idx, 1] = dst_tx_idx - ix_idx_

        # round offsets since we cannot make use of subpixel precision in
        # coarse levels (VERY IMPORTANT!!!)
        offset = torch.round(offset)

        return offset

    '''
    Matching step inside a pyramid scale
    '''
    def step_in_pyram(self, level, init_offset, src, dst):
        tile_size = self.cfg.TILE_SIZE
        n, num_tile_h, num_tile_w, _ = init_offset.size()
        ty_idx, tx_idx = utils.get_matrix_indices(num_tile_h, num_tile_w)
        sy_idx, sx_idx = self.idx_im(ty_idx), self.idx_im(tx_idx)
        search_radius = self.cfg.SEARCH_RADIUS
        padd = 1 # for subpixel fitting
        search_size = 2 * (search_radius + padd) + 1

        # tile appearance distance
        adist_map = self.tile_match(src, dst, init_offset) # N, H, W, S^2
        adist_map = adist_map.reshape(n, num_tile_h, num_tile_w, search_size, search_size)

        # spatial distance map
        sdist_map = self.sdist_map.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        sdist_map = sdist_map.expand(n, num_tile_h, num_tile_w, -1, -1).to(src.device)

        # total distance map
        dist_map_ = adist_map + self.cfg.SPATIAL_WEIGHT * sdist_map
        dist_map = dist_map_[:, :, :, padd:-padd, padd:-padd] # cancel the padding

        del adist_map, sdist_map

        # get translation
        dist_map = dist_map.reshape(n, num_tile_h, num_tile_w, -1)
        min_dist, peak_idx = torch.min(dist_map, dim=3)
        peak_idx = peak_idx.float()

        peak_x = torch.fmod(peak_idx, 2 * search_radius + 1)
        peak_y = torch.div(peak_idx - peak_x, 2 * search_radius + 1)

        # add initial offset into translation
        offset = init_offset.clone()
        offset[:, :, :, 0] = init_offset[:, :, :, 0] + peak_y - search_radius
        offset[:, :, :, 1] = init_offset[:, :, :, 1] + peak_x - search_radius

        peak_idx = torch.stack([peak_y+padd, peak_x+padd], dim=3).float()
        # y: [n, 3, 3, num_tile_h*num_tile_w]
        y = self.region_select(dist_map_, peak_idx)

        del dist_map_

        # prepare filters
        f_a11 = self.f_a11.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n, num_tile_h, num_tile_w, -1, -1).to(src.device)
        f_a12 = self.f_a12.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n, num_tile_h, num_tile_w, -1, -1).to(src.device)
        f_a21 = self.f_a21.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n, num_tile_h, num_tile_w, -1, -1).to(src.device)
        f_a22 = self.f_a22.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n, num_tile_h, num_tile_w, -1, -1).to(src.device)
        f_b1 = self.f_b1.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n, num_tile_h, num_tile_w, -1, -1).to(src.device)
        f_b2 = self.f_b2.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n, num_tile_h, num_tile_w, -1, -1).to(src.device)
        f_c = self.f_c.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(n, num_tile_h, num_tile_w, -1, -1).to(src.device)

        # See Eqn. (23)~(33) in the supplementary material of HDR+ paper
        f_a11 = (f_a11 * y).sum(dim=3).sum(dim=3, keepdim=True)
        f_a12 = (f_a12 * y).sum(dim=3).sum(dim=3, keepdim=True)
        f_a21 = (f_a21 * y).sum(dim=3).sum(dim=3, keepdim=True)
        f_a22 = (f_a22 * y).sum(dim=3).sum(dim=3, keepdim=True)
        f_b1 = (f_b1 * y).sum(dim=3).sum(dim=3, keepdim=True)
        f_b2 = (f_b2 * y).sum(dim=3).sum(dim=3, keepdim=True)
        f_c = (f_c * y).sum(dim=3).sum(dim=3, keepdim=True)

        del y

        f_a11 = F.relu(f_a11)
        f_a22 = F.relu(f_a22)
        det_a = f_a11 * f_a22 - f_a12**2
        f_a12 = torch.where(det_a < 0, torch.zeros_like(f_a12), f_a12)
        f_a21 = torch.where(det_a < 0, torch.zeros_like(f_a21), f_a21)

        mu_x = torch.div(-(f_a22 * f_b1 - f_a12 * f_b2), det_a)
        mu_y = torch.div(-(f_a11 * f_b2 - f_a12 * f_b1), det_a)
        s = f_c - 0.5 * (f_a11 * (mu_y**2) + 2 * f_a12 * mu_y * mu_x + f_a22 * (mu_x**2))

        del f_a11, f_a12, f_a21, f_a22, f_b1, f_b2, f_c

        mu_len = torch.sqrt(mu_y**2 + mu_x**2)
        dx = torch.where(mu_len < 1.0, mu_x, torch.zeros_like(mu_x))
        dy = torch.where(mu_len < 1.0, mu_y, torch.zeros_like(mu_y))

        offset[:, :, :, 0] = offset[:, :, :, 0] + dy[:, :, :, 0]
        offset[:, :, :, 1] = offset[:, :, :, 1] + dx[:, :, :, 0]

        del mu_x, mu_y, mu_len, dx, dy

        return offset, min_dist

    '''
    Input: two list of image tensors, representing the pyramid of the source
           and target images.
    Output:
        1. offset field of shape [N, H, W, 2], where H, W are image size and
           the last dimension is the vertial + horizontal offset.
        2. Distance map of shape [N, H, W].
    '''
    def align_pyramid(self, pyram_src, pyram_dst):
        # set useful fields
        num_levels = len(pyram_src)
        self.pyram_sizes = np.zeros((num_levels, 2))
        for _ in range(len(pyram_src) - 1, -1, -1):
            self.pyram_sizes[_] = pyram_dst[_].shape[2:]

        # hierarchical alignment
        offset_pyram = [] # a list to hold offsets at each level
        for level in range(num_levels - 1, -1, -1):
            src = pyram_src[level]
            dst = pyram_dst[level]
            if level == num_levels - 1: # zero offset at the beginning
                n, _, ih, iw = src.size()
                num_tiles_h = self.image_size_to_tile_num(ih)
                num_tiles_w = self.image_size_to_tile_num(iw)
                init_offset = torch.zeros(n, num_tiles_h, num_tiles_w, 2).to(src.device)
            else: # inherit offset from previous level
                init_offset = self.inherit_offset(level, offset_pyram[-1])
            offset, min_dist = self.step_in_pyram(level, init_offset, src, dst)
            offset_pyram.append(offset)
        offset_pyram.reverse()

        # upsample the offset to original size
        tile_offsets = offset_pyram[0].permute(0, 3, 1, 2)
        # reorganize tile-based offsets to image-based ones
        pixel_offsets = F.upsample(tile_offsets, scale_factor=self.cfg.TILE_SIZE, mode='nearest')

        del tile_offsets

        return pixel_offsets

    def forward(self, img_ref, img_alt_list):
        # Pad inputs to be divisible by tile size at the coarest level
        num_levels = self.cfg.NUM_LEVELS
        ds_factor = self.cfg.DS_FACTOR
        tile_size = self.cfg.TILE_SIZE
        multiplier = ds_factor ** (num_levels - 1) * tile_size
        _, _, ih, iw = img_ref.size()
        img_ref_padd = utils.img_modpad(img_ref, multiplier)
        img_alt_padd_list = []
        for _ in range(len(img_alt_list)):
            img_alt_padd_list.append(utils.img_modpad(img_alt_list[_], multiplier))
        _, _, oh, ow = img_ref_padd.size()

        # Build gaussian pyramid and perform alignment
        pyram_ref = self.build_pyramid(img_ref_padd)
        offset_list = []
        for _ in range(len(img_alt_padd_list)):
            pyram_alt = self.build_pyramid(img_alt_padd_list[_])
            offset = self.align_pyramid(pyram_alt, pyram_ref)
            del pyram_alt
            # cancel the padding
            ph, pw = (oh - ih) // 2, (ow - iw) // 2
            offset = offset[:, :, ph:ph+ih, pw:pw+iw].contiguous()
            offset_list.append(offset)

        del pyram_ref, img_alt_padd_list

        return offset_list

    def backward(self, pyram_src, pyram_dst):
        print('%s: this module does not support back propagation' %(self.__name__), file=sys.stderr)
        raise

