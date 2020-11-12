import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
from math import floor, atan2, pi, isnan, sqrt,copysign
import cv2
import pytorch_ssim

import lpips

#### gradient loss
def grad_sobel_loss(recon, gt, train=True):
    # sobel filter
    a=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().cuda().unsqueeze(0).unsqueeze(0)) 
    b=np.array([[-1, -2, -1],[0,0,0],[1,2,1]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=0, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().cuda().unsqueeze(0).unsqueeze(0))

    recon_h = conv1(recon)
    recon_v = conv2(recon)
    gt_h = conv1(gt)
    gt_v = conv2(gt)

    if train:
        loss = torch.mean(torch.abs(recon_h-gt_h) + torch.abs(recon_v-gt_v))
    else:
        loss = torch.mean(torch.abs(recon_h-gt_h) + torch.abs(recon_v-gt_v))
        loss = loss.data.item()
    
    return loss

class L1L2Loss(object):
    def __init__(self, l2ratio = 1.0):
        self.l2ratio=l2ratio
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def __call__(self, recon, gt):
        return self.l1loss(recon, gt) + self.l2ratio * self.l2loss(recon, gt)



class SMAPE(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma
        
    def __call__(self, recon, gt):
        numerator = torch.abs(recon - gt)
        denominator = torch.abs(recon) + torch.abs(gt) + self.sigma
        return torch.mean(numerator / denominator)

class L1GradL2Intensity(object):
    """weighted sum of L1 distance of gradient and L2 distance of pixel intensity
    Reference: CVPR'18 Burst Denoising with Kernel Prediction Networks 
    """
    def __init__(self, l2ratio=1.0):
        self.l2ratio = l2ratio
        a=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        self.conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv1.weight=nn.Parameter(torch.from_numpy(a).float().cuda().unsqueeze(0).unsqueeze(0)) 
        b=np.array([[-1, -2, -1],[0,0,0],[1,2,1]])
        self.conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1,padding=0, bias=False)
        self.conv2.weight=nn.Parameter(torch.from_numpy(b).float().cuda().unsqueeze(0).unsqueeze(0))
        self.l1loss = nn.L1Loss()
        self.l2loss = nn.MSELoss()

    def __call__(self, recon, gt):
        recon_h = self.conv1(recon)
        recon_v = self.conv2(recon)
        gt_h = self.conv1(gt)
        gt_v = self.conv2(gt)
        return self.l1loss(recon_h, gt_h) + self.l1loss(recon_v, gt_v) \
            + self.l2ratio * self.l2loss(recon, gt)

class AsymmetricL1(object):
    """ Asymmetric loss
    Reference: SIGGRAPH'19 Denoising with Kernel Prediction and Assymetric Loss Function
    To avoid oversmoothing
    asymmetricSlope > 1
    (asymmetricSlope = 1 <==> the normal L1 loss)
    """
    def __init__(self, asymmetricSlope=1.5):
        self.asymmetricSlope = asymmetricSlope
    
    def __call__(self, recon, gt, noisy):
        weightMap = (torch.sign(gt - recon) * torch.sign(recon-noisy) + 1.0) \
            * (self.asymmetricSlope - 1) / 2.0 + 1.0
        return torch.mean(weightMap * torch.abs(recon - gt))

class AsymmetricL2(object):
    def __init__(self, asymmetricSlope=1.5):
        self.asymmetricSlope = asymmetricSlope
    
    def __call__(self, recon, gt, noisy):
        weightMap = (torch.sign(gt - recon) * torch.sign(recon-noisy) + 1.0) \
            * (self.asymmetricSlope - 1) / 2.0 + 1.0
        return torch.mean(weightMap * (recon - gt)**2)

class AsymmetricL1L2(object):
    def __init__(self, asymmetricSlope=1.5, l2ratio = 1.0):
        self.l2ratio = l2ratio
        self.asymmetricSlope = asymmetricSlope
    
    def __call__(self, recon, gt, noisy):
        weightMap = (torch.sign(gt - recon) * torch.sign(recon-noisy) + 1.0) \
            * (self.asymmetricSlope - 1) / 2.0 + 1.0
        return self.l2ratio * torch.mean(weightMap * (recon-gt)**2) + \
            torch.mean(weightMap * torch.abs(recon - gt))

class AsymmetricSMAPE(object):
    def __init__(self, asymmetricSlope=1.5, sigma=0.01):
        self.asymmetricSlope = asymmetricSlope
        self.sigma = sigma
        
    def __call__(self, recon, gt, noisy):
        weightMap = (torch.sign(gt - recon) * torch.sign(recon-noisy) + 1.0) \
            * (self.asymmetricSlope - 1) / 2.0 + 1.0
        numerator = torch.abs(recon - gt)
        denominator = torch.abs(recon) + torch.abs(gt) + self.sigma
        return torch.mean(weightMap * (numerator / denominator))

class AsymmetricDownsampleL1(object):
    def __init__(self, asymmetricSlope=1.5, downsampleScale=0.5):
        self.asymmetricSlope = asymmetricSlope
        self.downsampleScale = downsampleScale

    def __call__(self, recon, gt, noisy):
        drecon = F.interpolate(recon, scale_factor=self.downsampleScale, mode='bilinear')
        dgt = F.interpolate(gt, scale_factor=self.downsampleScale, mode='bilinear')
        dnoisy = F.interpolate(noisy, scale_factor=self.downsampleScale, mode='bilinear')

        weightMap = (torch.sign(dgt - drecon) * torch.sign(drecon-dnoisy) + 1.0) \
            * (self.asymmetricSlope - 1) / 2.0 + 1.0

        return torch.mean(weightMap * torch.abs(drecon - dgt)) + nn.L1Loss()(recon, gt)


def grad_patch(patch_x, patch_y):
    gx = patch_x.ravel()
    gy = patch_y.ravel()
    G = np.vstack((gx, gy)).T
    x = np.dot(G.T, G)
    w, v = np.linalg.eig(x)
    index = w.argsort()[::-1]
    w = w[index]
    v = v[:, index]
    theta = atan2(v[1, 0], v[0, 0])
    if theta < 0:
        theta = theta + pi
    theta = theta/pi
    lamda = sqrt(abs(w[0]))
    u = (sqrt(abs(w[0])) -sqrt(abs(w[1]))) / ((sqrt(abs(w[0]))) + sqrt(abs(w[1])) + 0.00000000000000001)

    return theta, lamda, u




class WeightedTripleLoss(object):
    def __init__(self,temperature,h_hsize, l1_const,l2ratio, ssim_size):
        self.temperature = temperature
        self.h_hsize =h_hsize
        self.l1_const = l1_const
        self.l2_scale = l2ratio
        self.ssim_loss = pytorch_ssim.SSIM(window_size=ssim_size)

    def __call__(self, recon, gt, strs,cohers):

        (_,_, H, W) = gt.shape
        total_loss = 0.0
        for i1 in range(self.h_hsize, H - self.h_hsize, 1):

            for j1 in range(self.h_hsize, W - self.h_hsize, 1):
                idx1 = (slice(0,gt.shape[0]),slice(0,1),slice(i1 - self.h_hsize, i1 + self.h_hsize + 1), slice(j1 - self.h_hsize, j1 + self.h_hsize + 1))
                patch_gt = gt[idx1]
                patch_recon = recon[idx1]
                w_losses = torch.nn.functional.softmax(torch.stack(
                    (cohers[:,0,i1,j1] * self.l2_scale, strs[:,0,i1,j1],
                     torch.ones(strs.shape[0]).to(patch_recon.device) * self.l1_const),dim=-1) / self.temperature, dim=-1)
                w_losses = w_losses.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(patch_recon.device)
                weighted_l1 = (w_losses[:, 2]+0.1) * torch.abs(patch_recon - patch_gt)
                weighted_l2 = w_losses[:, 0] * ((patch_recon - patch_gt) ** 2)
                weighted_ssim = w_losses[:, 1] * (1 - self.ssim_loss(patch_recon, patch_gt))
                weighted_loss = weighted_l1 + weighted_l2 +weighted_ssim
                total_loss+=torch.mean(weighted_loss)

        return total_loss



class TripleAsymLoss(object):
    def __init__(self,temperature,h_hsize, l1_const,l2ratio, ssim_size,asymmetricSlope=1.25, downsampleScale=0.5):
        self.temperature = temperature
        self.h_hsize =h_hsize
        self.l1_const = l1_const
        self.l2_scale = l2ratio
        self.ssim_loss = pytorch_ssim.SSIM(window_size=ssim_size)
        self.asymmetricSlope = asymmetricSlope
        self.downsampleScale = downsampleScale

    def __call__(self, recon, gt, noisy, strs,cohers):

        (_,_, H, W) = gt.shape
        total_loss = 0.0
        for i1 in range(self.h_hsize, H - self.h_hsize, 1):

            for j1 in range(self.h_hsize, W - self.h_hsize, 1):

                idx1 = (slice(0,gt.shape[0]),slice(0,1),slice(i1 - self.h_hsize, i1 + self.h_hsize + 1), slice(j1 - self.h_hsize, j1 + self.h_hsize + 1))
                patch_gt = gt[idx1]
                patch_recon = recon[idx1]

                w_losses = torch.nn.functional.softmax(torch.stack(
                    (cohers[:,0,i1,j1] * self.l2_scale, strs[:,0,i1,j1],
                     torch.ones(strs.shape[0]).to(patch_recon.device) * self.l1_const),dim=-1) / self.temperature, dim=-1)
                w_losses = w_losses.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(patch_recon.device)
                weighted_l1 = w_losses[:, 2] * torch.abs(patch_recon - patch_gt)
                weighted_l2 = w_losses[:, 0] * ((patch_recon - patch_gt) ** 2)
                weighted_ssim = w_losses[:, 1] * (1 - self.ssim_loss(patch_recon, patch_gt))
                weighted_loss = weighted_l1 + weighted_l2 +weighted_ssim
                total_loss+=torch.mean(weighted_loss)
        drecon = F.interpolate(recon, scale_factor=self.downsampleScale, mode='bilinear')
        dgt = F.interpolate(gt, scale_factor=self.downsampleScale, mode='bilinear')
        dnoisy = F.interpolate(noisy, scale_factor=self.downsampleScale, mode='bilinear')

        weightMap = (torch.sign(dgt - drecon) * torch.sign(drecon - dnoisy) + 1.0) \
                    * (self.asymmetricSlope - 1) / 2.0 + 1.0
        asym_down_l1 = torch.mean(weightMap * torch.abs(drecon - dgt)) + nn.L1Loss()(recon, gt)

        return 0.8*total_loss + 0.2*asym_down_l1


class HardTripleLoss(object):
    def __init__(self,temperature,h_hsize, ssim_thres,l2_thres, ssim_size):
        self.temperature = temperature
        self.h_hsize =h_hsize
        self.l1_const = ssim_thres
        self.l2_scale = l2_thres
        self.ssim_loss = pytorch_ssim.SSIM(window_size=ssim_size)

    def __call__(self, recon, gt, strs,cohers):

        (_,_, H, W) = gt.shape
        total_loss = 0.0
        for i1 in range(self.h_hsize, H - self.h_hsize, 1):

            for j1 in range(self.h_hsize, W - self.h_hsize, 1):
                idx1 = (slice(0,gt.shape[0]),slice(0,1),slice(i1 - self.h_hsize, i1 + self.h_hsize + 1), slice(j1 - self.h_hsize, j1 + self.h_hsize + 1))
                patch_gt = gt[idx1]
                patch_recon = recon[idx1]
                weight1 = (strs[:,0,i1,j1]>self.l2_scale).float()
                weight2 = (cohers[:, 0, i1, j1] > self.l1_const).float()
                weight3 = ((weight1+weight2)==0).float()
                w_losses = torch.stack(
                    (weight1,weight2,weight3),dim=-1)
                w_losses = w_losses.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(patch_recon.device)
                weighted_l1 = w_losses[:, 2] * torch.abs(patch_recon - patch_gt)
                weighted_l2 = w_losses[:, 0] * ((patch_recon - patch_gt) ** 2)
                weighted_ssim = w_losses[:, 1] * (1 - self.ssim_loss(patch_recon, patch_gt))
                weighted_loss = weighted_l1 + weighted_l2 +weighted_ssim
                total_loss+=torch.mean(weighted_loss)

        return total_loss

