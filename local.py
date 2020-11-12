import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalStructureAnalyse(nn.Module):
    def __init__(self):
        super(LocalStructureAnalyse, self).__init__()
        self.weight = torch.Tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]]).view((1, 1, 3, 3)) / 16.
        self.weight = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).view((1, 1, 3, 3)) / 9.

    '''
    Input: NCHW image tensor, where C must equal 1
    Output: eigen values and vectors of shape NHW
    '''
    def forward(self, x):
        n, c, h, w = x.size()

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
        A = A.view(n, h, w, 2, 2)

        lambda1, lambda2, e11, e12, e21, e22 = eig2x2(A)

        return lambda1, lambda2, e11, e12, e21, e22

    
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


if __name__ == '__main__':
    data = np.load('data/AAPM/patient56789.npz')
    img = torch.from_numpy(data['FD'][332,:,:,:]).float().unsqueeze(0)
    lambda1, lambda2, e11, e12, e21, e22 = LocalStructureAnalyse()(img)

    diff = torch.sqrt(lambda2) - torch.sqrt(lambda1)
    coherence = (torch.sqrt(lambda2) - torch.sqrt(lambda1)) / (
        torch.sqrt(lambda2) + torch.sqrt(lambda1) + 1e-5)
    angle = torch.atan(e22 / e21)

    from matplotlib import pyplot as plt
    plt.imshow(angle[0,:,:].numpy(), cmap='gray')#, vmin=0.05, vmax=0.1)
    plt.show()
    # npimg = coherence[0,:,:].numpy().flatten()
    # print(npimg[np.argsort(npimg, axis=None)[-1000:]])


