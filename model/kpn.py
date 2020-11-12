"""
The network structure I use is derived from
SIGGRAPH'18 Denoising with Kernel Prediction and Asymmetric Loss Functions
http://drz.disneyresearch.com/~jnovak/publications/KPAL/index.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import numpy as np
from matplotlib import pyplot as plt

from .kpn_util import LocalConv2d_No


class ResidueBlock(nn.Module):
    def __init__(self, filterChannels, kernelSize):
        super(ResidueBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(filterChannels, filterChannels, kernelSize, padding=1)
        self.conv2 = nn.Conv2d(filterChannels, filterChannels, kernelSize, padding=1)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out

class DilResBlock(nn.Module):
    def __init__(self, filterChannels, kernelSize):
        super(DilResBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = DilConv(filterChannels, filterChannels)
        self.conv2 = nn.Conv2d(filterChannels, filterChannels, kernelSize, padding=1)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out

class ResidueBlockS(nn.Module):
    def __init__(self, filterChannels, kernelSize):
        super(ResidueBlockS, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(filterChannels, filterChannels, kernelSize, padding=1, groups=2)
        self.conv2 = nn.Conv2d(filterChannels, filterChannels, kernelSize, padding=1, groups=2)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        return out




class KPN(nn.Module):
    def __init__(self, netOpt):
        super(KPN, self).__init__()
        self.netOpt = netOpt
        self.sourceEncoder = nn.Sequential(
            nn.Conv2d(1, netOpt['filterChannels'], 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(netOpt['filterChannels'], netOpt['filterChannels'], 3, padding=1)
        )
        featureExtractorList = []
        for _ in range(netOpt['numBlocks']):
            featureExtractorList.append(
                ResidueBlock(netOpt['filterChannels'], 3)
            )
        self.featureExtractor=nn.Sequential(*featureExtractorList)

        finalFilterChannels = netOpt['KPNKernelSize'] ** 2
        self.kernelPredictor = nn.Sequential(
                nn.Conv2d(netOpt['filterChannels'], finalFilterChannels, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(finalFilterChannels, finalFilterChannels, 1)
        )

        self.localConv = LocalConv2d_No(
            in_channels=1,
            out_channels=1,
            kernel_size=netOpt['KPNKernelSize'],
            bias=False
        )


    def forward(self, x, verbose=False,temperature=1.0):
        features = self.sourceEncoder(x)
        features = self.featureExtractor(features)
        kernels = self.kernelPredictor(features)
        kernels = kernels / self.temp
        kernels  = kernels / temperature
        if 'kernelWeightSoftmax' in self.netOpt:
            if self.netOpt['kernelWeightSoftmax']:
                kernels = F.softmax(kernels, dim=1)
        else:
            kernels = F.softmax(kernels, dim=1)
            
        out = self.localConv.forward(x,kernels)
        if verbose:
            return out, kernels
        else:
            return out


class KPNSlim(nn.Module):
    def __init__(self, netOpt):
        super(KPNSlim, self).__init__()
        self.netOpt = netOpt
        self.sourceEncoder = nn.Sequential(
            nn.Conv2d(1, netOpt['filterChannels'], 3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(netOpt['filterChannels'], netOpt['filterChannels'], 3, padding=1)
        )

        featureExtractorList = []
        for _ in range(netOpt['numBlocks']):
            featureExtractorList.append(
                ResidueBlockS(netOpt['filterChannels'], 3)
            )
        self.featureExtractor = nn.Sequential(*featureExtractorList)

        finalFilterChannels = netOpt['KPNKernelSize'] ** 2
        self.kernelPredictor = nn.Sequential(
            nn.Conv2d(netOpt['filterChannels'], finalFilterChannels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(finalFilterChannels, finalFilterChannels, 1)
        )

        self.localConv = LocalConv2d_No(
            in_channels=1,
            out_channels=1,
            kernel_size=netOpt['KPNKernelSize'],
            bias=False
        )

    def forward(self, x, verbose=False, temperature=1.0):
        features = self.sourceEncoder(x)
        features = self.featureExtractor(features)
        kernels = self.kernelPredictor(features)
		kernels = kernels / self.temp
        kernels = kernels / temperature
        if 'kernelWeightSoftmax' in self.netOpt:
            if self.netOpt['kernelWeightSoftmax']:
                kernels = F.softmax(kernels, dim=1)
        else:
            kernels = F.softmax(kernels, dim=1)

        out = self.localConv.forward(x, kernels)
        if verbose:
            return out, kernels
        else:
            return out

class DilConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DilConv, self).__init__()
        channels = out_channel // 4
        self.conv1 = nn.Conv2d(in_channel, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channel, channels, 3, padding=(2,2),dilation=(2,2))
        self.conv3 = nn.Conv2d(in_channel, channels, 3, padding=(2, 3), dilation=(2, 3))
        self.conv4 = nn.Conv2d(in_channel, channels, 3, padding=(3, 2), dilation=(3, 2))

    def forward(self,x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        return(torch.cat([conv1,conv2,conv3,conv4],dim=1))



if __name__=='__main__':
    

    #### Overfit one images
    import scipy.io
    matFile = scipy.io.loadmat('data/AAPMTest/AAPMTest.mat')
    lows = matFile['lows']
    highs = matFile['highs']

    import dataset
    ctDataset = dataset.LowDoseCTDataset(lows, highs)
    dataLoader = torch.utils.data.DataLoader(ctDataset, batch_size=1, 
        shuffle=True, num_workers=2, pin_memory=True)
    
    import initialize
    args = initialize.InitParams()
    opt = initialize.InitParamsYaml()
    model = KPN(opt['network'])
    model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                weight_decay=args.weightDecay, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80], gamma=0.5)

    if args.evaluate:
        model.load_state_dict(torch.load(args.ckptDir+'/test.pth'))
    else:
        if args.resume:
            model.load_state_dict(torch.load(args.ckptDir+'/test.pth'))
        model.train()
        for epoch in range(args.startEpoch, args.startEpoch + args.batchSize):
            scheduler.step()
            losses=[]
            for i, (low, high) in enumerate(dataLoader):
                low = low.cuda()
                high = high.cuda()
                output = model.forward(low)
                loss = criterion(output, high)
                losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Loss of the epoch {}: {}'.format(epoch, sum(losses)/len(losses)))    

        torch.save(model.state_dict(),args.ckptDir+'/test.pth')
    

    model.eval()
    import skimage
    psnrs = []
    ssims = []

    dataLoader = torch.utils.data.DataLoader(ctDataset, batch_size=1, shuffle=False)
    for i, (low, high) in enumerate(dataLoader):
        low = low.cuda()
        high = high.cuda()
        output = model.forward(low)

        fig = plt.figure()
        high = high.detach().cpu().numpy()
        high = np.squeeze(high)
        plt.imshow(high, cmap='gray')
        fig.savefig(args.debugDir + '/{}_high.png'.format(i))
        
        fig = plt.figure()
        low = low.detach().cpu().numpy()
        low = np.squeeze(low)
        plt.imshow(low, cmap='gray')
        fig.savefig(args.debugDir + '/{}_low.png'.format(i))
        
        fig = plt.figure()
        output = output.detach().cpu().numpy()
        output = np.squeeze(output)
        plt.imshow(output, cmap='gray')
        fig.savefig(args.debugDir + '/{}_recon.png'.format(i))

        psnr = skimage.measure.compare_psnr(high, output)
        ssim = skimage.measure.compare_ssim(high, output)
        psnrs.append(psnr)
        ssims.append(ssim)
        print('image {}\t psnr: {}\t ssim: {}\t'.format(i,psnr,ssim))

    print('average psnr: {}\t average ssim: {}'.format(sum(psnrs)/len(psnrs), sum(ssims)/len(ssims)))