import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def rgb2yuv(x):
    r, g, b = x[:, 0, :, :], x[:, 1, :, :], x[:, 2, :, :]
    y = 0.257 * r + 0.504 * g + 0.098 * b + 16 / 255.
    u = 0.439 * r - 0.368 * g - 0.071 * b + 128 / 255.
    v = -0.148 * r - 0.291 * g + 0.439 * b + 128 / 255.
    y = torch.clamp(y, 0., 1.)
    u = torch.clamp(u, 0., 1.)
    v = torch.clamp(v, 0., 1.)
    yuv = torch.stack([y, u, v], dim=1)
    return yuv

def get_matrix_indices(h, w):
    h_idx_mat, w_idx_mat = np.meshgrid(np.arange(0, h), np.arange(0, w))
    h_idx, w_idx = h_idx_mat.flatten().astype(np.int32), w_idx_mat.flatten().astype(np.int32)
    return h_idx, w_idx

def img_modpad(img, multiplier, mode='constant'):
    n, c, h, w = img.size()
    oh = np.ceil(h * 1.0 / multiplier) * multiplier
    ow = np.ceil(w * 1.0 / multiplier) * multiplier
    oh, ow = int(oh), int(ow)
    h_padd, w_padd = (oh - h) // 2, (ow - w) // 2
    img = F.pad(img, (w_padd, ow-w-w_padd, h_padd, oh-h-h_padd), mode=mode)
    return img

def img_crop_with_pad(img, top, left, bottom, right):
    height, width = bottom - top, right - left
    n, ch, ih, iw = img.size()
    nt, nl = max(top, 0), max(left, 0)
    nb, nr = min(ih, bottom), min(iw, right)
    ct = 0 if top >= 0 else -top
    cb = ct + height if bottom <= ih else ih - top
    cl = 0 if left >= 0 else -left
    cr = cl + width if right <= iw else iw - left

    cropped = torch.zeros((n, ch, height, width), dtype=img.dtype).to(img.device)
    cropped[:, :, ct:cb, cl:cr] = img[:, :, nt:nb, nl:nr]

    return cropped

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_gaussian_kernel(window_size, sigma, channel=1):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def batched_index_select(input, dim, index):
    views = [input.shape[0]] + \
		[1 if i != dim else -1 for i in xrange(1, len(input.shape))]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    out = torch.gather(input, dim, index)
    return out

def batched_index_select_2d(v, r, c):
    # v: [N, H, W]
    # r, c: [N, IH, IW, K]
    # output: [N, IH, IW, K] values from v
    n, h, w = v.size()
    n, ih, iw, k = r.size()
    v = v.view(n, -1) # N, H*W
    r = r.view(n, -1) # N, IH*IW*K
    c = c.view(n, -1) # N, IH*IW*K
    lin_idx = r * w + c
    out = batched_index_select(v, 1, lin_idx) # out: N, IH*IW*K
    out = out.view(n, ih, iw, k)

    return out

'''
SVD decomposition of 2*2 matrices. Details referred to
https://scicomp.stackexchange.com/questions/8899/robust-algorithm-for-2-times-2-svd
'''
def svd2x2(x):
    # input: a multi-dimensional tensor with the last two dimensions equal to 2
    dims = list(x.shape)
    ndims = len(x.shape)
    x00 = x[..., 0, 0]
    x01 = x[..., 0, 1]
    x10 = x[..., 1, 0]
    x11 = x[..., 1, 1]

    E = (x00 + x11) * 0.5
    F = (x00 - x11) * 0.5
    G = (x10 + x01) * 0.5
    H = (x10 - x01) * 0.5
    Q = torch.sqrt(E**2 + H**2)
    R = torch.sqrt(F**2 + G**2)
    sx = Q + R
    sy = Q - R
    a1 = torch.atan2(G, F)
    a2 = torch.atan2(H, E)
    theta = (a2 - a1) * 0.5
    phi = (a2 + a1) * 0.5

    u = torch.stack([torch.cos(phi),
                     -torch.sin(phi),
                     torch.sin(phi),
                     torch.cos(phi)],
                    dim=ndims-2).view(*(dims[:-2] + [2, 2]))
    s = torch.stack([sx,
                     torch.zeros_like(sx),
                     torch.zeros_like(sy),
                     sy],
                    dim=ndims-2).view(*(dims[:-2] + [2, 2]))
    v = torch.stack([torch.cos(theta),
                     -torch.sin(theta),
                     torch.sin(theta),
                     torch.cos(theta)],
                    dim=ndims-2).view(*(dims[:-2] + [2, 2]))

    # x = u * s * v
    return u, s, v

'''
Pesudo inverse of 2*2 matrices
'''
def pinv2x2(x):
    # input: a multi-dimensional tensor with the last two dimensions equal to 2
    u, s, v = svd2x2(x)
    dims = x.shape
    ndims = len(dims)
    u_t = u.permute(*(range(ndims-2) + [ndims-1, ndims-2]))
    v_t = v.permute(*(range(ndims-2) + [ndims-1, ndims-2]))
    s_inv = torch.where(torch.le(torch.abs(s), 1e-6), s, 1.0 / (s + 1e-6))
    x_pinv = torch.matmul(v_t, torch.matmul(s_inv, u_t))
    return x_pinv

'''
Eigen analysis of 2*2 matrices
'''
def eig2x2(x):
    # input: a multi-dimensional tensor with the last two dimensions equal to 2

    # compute eigen vectors and eigenvalues, following
    # www.math.harvard.edu/archive/21b_fall_04/exhibits/2dmatrices/index.html
    T = x[..., 0, 0] + x[..., 1, 1]
    D = x[..., 0, 0] * x[..., 1, 1] - x[..., 0, 1] * x[..., 1, 0]
    lambda1 = T * 0.5 + torch.sqrt(torch.abs(T * T * 0.25 - D))
    lambda2 = T * 0.5 - torch.sqrt(torch.abs(T * T * 0.25 - D))

    e11 = lambda1 - x[..., 1, 1]
    e12 = x[..., 1, 0]
    e21 = lambda2 - x[..., 1, 1]
    e22 = x[..., 1, 0]

    e11 = torch.where(torch.abs(x[..., 1, 0]) < 1e-9, torch.ones_like(e11), e11)
    e12 = torch.where(torch.abs(x[..., 1, 0]) < 1e-9, torch.zeros_like(e12), e12)
    e21 = torch.where(torch.abs(x[..., 1, 0]) < 1e-9, torch.zeros_like(e21), e21)
    e22 = torch.where(torch.abs(x[..., 1, 0]) < 1e-9, torch.ones_like(e22), e22)

    # swap to keep lambda1 as the smaller eigen values
    lamb1_abs = torch.abs(lambda1)
    lamb2_abs = torch.abs(lambda2)
    lambda1_ = torch.where(torch.lt(lamb1_abs, lamb2_abs), lambda1, lambda2)
    lambda2_ = torch.where(torch.lt(lamb1_abs, lamb2_abs), lambda2, lambda1)
    e11_ = torch.where(torch.lt(lamb1_abs, lamb2_abs), e11, e21)
    e12_ = torch.where(torch.lt(lamb1_abs, lamb2_abs), e12, e22)
    e21_ = torch.where(torch.lt(lamb1_abs, lamb2_abs), e21, e11)
    e22_ = torch.where(torch.lt(lamb1_abs, lamb2_abs), e22, e12)

    lambda1_, labmda2_ = F.relu(lambda1_), F.relu(lambda2_)

    return lambda1_, lambda2_, e11_, e12_, e21_, e22_
