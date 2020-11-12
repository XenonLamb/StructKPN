import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numba as nb
from scipy.special import softmax
from math import floor, atan2, pi, isnan, sqrt,copysign
import cv2


def patch_process(img_gx, img_gy, h_hsize=4, step=1, l1_const=0.1, temperature=1.0):
    H, W = img_gx.shape
    #patch_num = len(range(h_hsize, H - h_hsize, step)) * len(range(h_hsize, W - h_hsize, step))
    strs = np.zeros((H, W))
    cohers = np.zeros((H, W))
    count = 0
    id1 = 0
    id2 = 0
    for i1 in range(h_hsize, H - h_hsize, step):

        for j1 in range(h_hsize, W - h_hsize, step):
            # pos = i1 % types_size * types_size + j1 % types_size
            # if pos in [0, 3, 5, 10, 12, 15]:
            #     continue
            idx1 = (slice(i1 - h_hsize, i1 + h_hsize + 1), slice(j1 - h_hsize, j1 + h_hsize + 1))
            patchX = img_gx[idx1]
            patchY = img_gy[idx1]
            strength, coherence = grad_patch(patchX, patchY)

            strs[i1, j1] = strength
            cohers[i1,j1] = coherence
            id2 += 1
            id2 = id2 % len(range(h_hsize, W - h_hsize, step))
            # if strength > 0.01 or coherence > 0.055:
            #     continue
            # assert count == patch_num
        id1 += 1
    return strs, cohers

def _compute_grads(highs, h_hsize=4, l1_const = 0.2, tempterature=1.0,splitname="train"):
    dstrs = []
    dcohers = []
    for i in range(highs.shape[0]):
        im = highs[i, 0, :, :]
        #im = im.astype('float') / 255.
        gx, gy = np.gradient(im)
        strs, cohers = patch_process(gx, gy, h_hsize=h_hsize, l1_const=l1_const, temperature=tempterature)
        dstrs.append(strs)
        dcohers.append(cohers)

    dstrs = np.array(dstrs)
    dcohers = np.array(dcohers)
    np.savez(
        # np.savez_compressed(
        ("./data/AAPM/grads_"+splitname+".npz"),
        strs=dstrs, cohs=dcohers)


    return dstrs, dcohers

def _compute_weight(strs, cohers, h_hsize=4, step=1, l1_const=0.1, temperature=1.0):
    H, W = strs.shape
    #patch_num = len(range(h_hsize, H - h_hsize, step)) * len(range(h_hsize, W - h_hsize, step))
    w_l1 = np.zeros((H, W))
    w_l2 = np.zeros((H, W))
    w_ssim = np.zeros((H, W))
    count = 0
    id1 = 0
    id2 = 0
    for i1 in range(h_hsize, H - h_hsize, step):

        for j1 in range(h_hsize, W - h_hsize, step):
            # pos = i1 % types_size * types_size + j1 % types_size
            # if pos in [0, 3, 5, 10, 12, 15]:
            #     continue
            idx1 = (slice(i1 - h_hsize, i1 + h_hsize + 1), slice(j1 - h_hsize, j1 + h_hsize + 1))
            strength = strs[i1, j1]
            coherence = cohers[i1, j1]
            weights = softmax((np.array([strength, coherence, l1_const])/temperature))
            w_l2[id1+h_hsize, id2+h_hsize] = weights[0]
            w_ssim[id1+h_hsize, id2+h_hsize] = weights[1]
            w_l1[id1+h_hsize, id2+h_hsize] = weights[2]
            id2 += 1
            id2 = id2 % len(range(h_hsize, W - h_hsize, step))
            # if strength > 0.01 or coherence > 0.055:
            #     continue
            # assert count == patch_num
        id1 += 1
    return w_l2, w_ssim, w_l1

def grad_patch(patch_x, patch_y):
    gx = patch_x.ravel()
    gy = patch_y.ravel()
    G = np.vstack((gx, gy)).T
    x = np.dot(G.T, G)
    w, v = np.linalg.eig(x)
    index = w.argsort()[::-1]
    w = w[index]
    v = v[:, index]
    lamda = sqrt(abs(w[0]))
    u = (sqrt(abs(w[0])) - sqrt(abs(w[1]))) / ((sqrt(abs(w[0]))) + sqrt(abs(w[1])) + 0.00000000000000001)
    # u = (copysign(1,w[0])*sqrt(abs(w[0])) - copysign(1,w[1])*sqrt(abs(w[1]))) / ((copysign(1,w[0])*sqrt(abs(w[0]))) + copysign(1,w[1])*sqrt(abs(w[1])) + 0.00000000000000001)
    return lamda, u


class LowDoseCTDataset(Dataset):


    def __init__(self, lows, highs, transforms=None, splitname="train"):
        super(LowDoseCTDataset, self).__init__()
        if len(lows.shape) == 3:
            lows = np.expand_dims(lows,1)
        if len(highs.shape) == 3:
            highs = np.expand_dims(highs,1)
        self.lows = lows
        self.highs = highs
        #dstrs, dcohers = _compute_grads(highs, splitname=splitname)
        npys = np.load(("./data/AAPM/grads_"+splitname+".npz"))
        self.dstrs = npys['strs']
        self.dcohers = npys['cohs']
        if self.dstrs.shape[0]!=lows.shape[0]:
            orig_shape = self.dstrs.shape
            orig_shape=(lows.shape[0],*(orig_shape[1:]))
            self.dstrs = np.zeros(orig_shape, dtype=self.dstrs.dtype)
            self.dcohers = np.zeros(orig_shape, dtype=self.dcohers.dtype)
        print("finish loading npys")
        #print(self.lows.shape)
        #print(self.dstrs.shape)
        #print(self.dcohers.shape)


        #self.dstrs = dstrs
        #self.dcohers = dcohers
        assert lows.shape[0] == highs.shape[0]
        self.transforms = transforms


    def __getitem__(self, index):
        low = self.lows[index,:,:,:]
        high = self.highs[index,:,:,:]
        #w_l2,w_ssim,w_l1 = _compute_weight(self.dstrs[index], self.dcohers[index])
        #w_l2 = np.expand_dims(w_l2,axis=0)
        #w_l1 = np.expand_dims(w_l1, axis=0)
        #w_ssim = np.expand_dims(w_ssim, axis=0)
        #print(low.shape)
        #print(w_l2.shape)
        #print(w_l1.shape)


        # low = torch.from_numpy(low).float()
        # high = torch.from_numpy(high).float()

        low, high, strs, cohers = self.transforms(low, high, np.expand_dims(self.dstrs[index],axis=0), np.expand_dims(self.dcohers[index],axis=0))
        #print(w_l2.shape)
        #print(w_l1.shape)
        #print(index)
        return low, high, strs, cohers
    
    def __len__(self):
        return self.lows.shape[0]

def CreateDataLoader(opt, trainDataset, testDataset):
    trainLoader = torch.utils.data.DataLoader(trainDataset,
        batch_size = opt['datasets']['train']['batchSize'],
        shuffle = opt['datasets']['train']['shuffle'],
        num_workers = opt['datasets']['train']['workers'],
        pin_memory = True
    )
    testLoader = torch.utils.data.DataLoader(testDataset,
        batch_size = opt['datasets']['test']['batchSize'],
        shuffle = opt['datasets']['test']['shuffle'],
        num_workers = opt['datasets']['test']['workers'],
        pin_memory = True
    )
    return trainLoader, testLoader


if __name__=='__main__':
    # import scipy.io
    # matFile = scipy.io.loadmat('data/AAPMTest/AAPMTest.mat')
    # lows = matFile['lows']
    # highs = matFile['highs']


    import initialize
    opt = initialize.InitParamsYamlDebug()

    import os
    testData = np.load(os.path.join(opt['datasets']['npyDir'], 
        opt['datasets']['train']['fileName']))

    from cotransforms import Option2Transforms
    transforms = Option2Transforms(opt, train=True)
    dataset = LowDoseCTDataset(testData['LD'], testData['FD'], transforms)
    

    from matplotlib import pyplot as plt
    for i in range(10):
        LD, FD, w_l2, w_l1, w_ssim = dataset.__getitem__(i)
        plt.imshow(FD[0], cmap='gray')
        plt.show()
        plt.imshow(LD[0], cmap='gray')
        plt.show()
    # print(LD-FD)
    # print((LD-FD).sum()/512.0/512.0)