# code copied heavily from FlowNetPytorch

from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage
from PIL import Image
import torchvision.transforms.functional as F

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''

def Option2Transforms(opt, train=True, reduce=False):
    transformList = []
    trainOpt = opt['datasets']['train']
    # training augmentation
    if train:
        trainOpt = opt['datasets']['train']
        if trainOpt['horizontalFlip']:
            transformList.append(RandomHorizontalFlip())
        if trainOpt['verticalFlip']:
            transformList.append(RandomVerticalFlip())
        if trainOpt['rot90']:
            transformList.append(RandomRot90())
        if trainOpt['randomScale']:
            transformList.append(RandomScale())
        if trainOpt['randomCrop']:
            transformList.append(RandomCrop(
                trainOpt['patchSize'], trainOpt['discardBlackPatches']))

    
    # toTensor
    transformList.append(Npy2Tensor())

    # Normalize
    normOpt = opt['datasets']['normalize']
    if normOpt['method'] == 'AlreadyNormalized':
        pass
    elif normOpt['method'] == 'Normalize':
        transformList.append(CoNormalize(
            FDMean=[normOpt['fdmean']], 
            FDStd=[normOpt['fdstd']], 
            LDMean=[normOpt['ldmean']], 
            LDStd=[normOpt['ldstd']]))
    elif normOpt['method'] == 'Clip2LungNormalize':
        transformList.append(Clip2LungNormalize())
    elif normOpt['method'] == 'ClipNormalize':
        transformList.append(ClipNormalize(
            lower = normOpt['lower'],
            upper = normOpt['upper']
        ))
    elif normOpt['method'] == 'DicomStandardNormalize':
        transformList.append(DicomStandardNormalize(
            c = normOpt['c'],
            w = normOpt['w'],
            RescaleSlope = normOpt['RescaleSlope'],
            RescaleIntercept = normOpt['RescaleIntercept']
        ))
    else:
        raise NotImplementedError
    
    # Suspect cropped patch has too much air, crop after normalization
    if train:
        if trainOpt['randomCropAfterNormalization']:
            assert not trainOpt['randomCrop']
            transformList.append(RandomCropAfterNormalization(
                trainOpt['patchSize'], trainOpt['discardBlackPatches']))
        if 'noiseReduce' in trainOpt:
            if trainOpt['noiseReduce']:
                transformList.append(NoiseReduce(
                    noiseReduceMaxLevel=trainOpt['noiseReduceMaxLevel']))
    elif reduce:
        if trainOpt['randomCropAfterNormalization']:
            assert not trainOpt['randomCrop']
            transformList.append(RandomCropAfterNormalization(
                trainOpt['patchSize'], trainOpt['discardBlackPatches']))

    return Compose(transformList)


class Compose(object):
    """ Composes several co_transforms together.
    For example:    
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, LD, FD, w1, w2):
        for t in self.co_transforms:
            LD,FD , w1, w2 = t(LD,FD, w1, w2)
        return LD,FD, w1, w2


class Npy2Tensor(object):
    """Converts a numpy.ndarray (1 x H x W) to a torch.FloatTensor of shape (1 x H x W)."""

    def __call__(self, LD, FD, w1, w2):
        assert(isinstance(LD, np.ndarray))
        assert(isinstance(FD, np.ndarray))
        LD = torch.from_numpy(LD)
        FD = torch.from_numpy(FD)
        w1 = torch.from_numpy(w1)
        w2 = torch.from_numpy(w2)
        #w3 = torch.from_numpy(w3)

        return LD.float(), FD.float(), w1.float(), w2.float()


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input,target):
        return self.lambd(input,target)


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, LD, FD, w1 ,w2):
        _, h1, ww1 = LD.shape
        th, tw = self.size
        x1 = int(round((ww1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))
        LD = LD[:, y1: y1 + th, x1: x1 + tw]
        FD = FD[:, y1: y1 + th, x1: x1 + tw]
        w1 = w1[:, y1: y1 + th, x1: x1 + tw]
        w2 = w2[:, y1: y1 + th, x1: x1 + tw]
        #w3 = w3[:, y1: y1 + th, x1: x1 + tw]
        return LD, FD, w1, w2

class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, target):
        h, w, _ = inputs[0].shape
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs,target
        if w < h:
            ratio = self.size/w
        else:
            ratio = self.size/h

        inputs[0] = ndimage.interpolation.zoom(inputs[0], ratio, order=self.order)
        inputs[1] = ndimage.interpolation.zoom(inputs[1], ratio, order=self.order)

        target = ndimage.interpolation.zoom(target, ratio, order=self.order)
        target *= ratio
        return inputs, target

class RandomScale(object):
    def __init__(self, order=2):
        self.order = order
        scale = np.random.uniform(low=1.0, high=1.5)
        enlargen = np.random.randint(2)
        if enlargen:
            self.scale = scale
        else:
            self.scale = 1.0 / scale
    
    def __call__(self, LD, FD):
        LDresize = ndimage.interpolation.zoom(LD[0,:,:], self.scale, order=self.order)
        FDresize = ndimage.interpolation.zoom(FD[0,:,:], self.scale, order=self.order)
        return LDresize[np.newaxis], FDresize[np.newaxis]


class RandomCrop(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Notice that if more than ${discardBlackPatches} parts are smaller than 24(air)
    then it will be recropped
    """

    def __init__(self, size, discardBlackPatches):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.discardBlackPatches = discardBlackPatches

    def __call__(self, LD, FD, w1, w2):
        c, h, w = LD.shape
        th, tw = self.size
        if w == tw and h == th:
            return LD, FD, w1, w2
        if self.discardBlackPatches == 1.0:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            LDPatch = LD[:, y1: y1 + th,x1: x1 + tw]
            FDPatch = FD[:, y1: y1 + th,x1: x1 + tw]
            w1Patch = w1[:, y1: y1 + th, x1: x1 + tw]
            w2Patch = w2[:, y1: y1 + th, x1: x1 + tw]

            #w3Patch = w3[:, y1: y1 + th, x1: x1 + tw]
        else:
            numAir = c * th * tw
            while numAir > self.discardBlackPatches * c * th * tw:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)

                LDPatch = LD[:, y1: y1 + th,x1: x1 + tw]
                FDPatch = FD[:, y1: y1 + th,x1: x1 + tw]
                w1Patch = w1[:, y1: y1 + th, x1: x1 + tw]
                w2Patch = w2[:, y1: y1 + th, x1: x1 + tw]
                #w3Patch = w3[:, y1: y1 + th, x1: x1 + tw]
                numAir = (FDPatch < 25).sum()
        return LDPatch, FDPatch, w1Patch,w2Patch

class RandomCropAfterNormalization(object):
    """Crops the given numpy array at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Notice that if more than ${discardBlackPatches} parts are smaller than 24(air)
    then it will be recropped
    """

    def __init__(self, size, discardBlackPatches):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.discardBlackPatches = discardBlackPatches

    def __call__(self, LD, FD,w1, w2):
        c, h, w = LD.shape
        th, tw = self.size
        if w == tw and h == th:
            return LD, FD
        if self.discardBlackPatches == 1.0:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            LDPatch = LD[:, y1: y1 + th,x1: x1 + tw]
            FDPatch = FD[:, y1: y1 + th,x1: x1 + tw]
            w1Patch = w1[:, y1: y1 + th, x1: x1 + tw]
            w2Patch = w2[:, y1: y1 + th, x1: x1 + tw]
            #w3Patch = w3[:, y1: y1 + th, x1: x1 + tw]
        else:
            numAir = c * th * tw
            while numAir > self.discardBlackPatches * c * th * tw:
                x1 = random.randint(0, w - tw)
                y1 = random.randint(0, h - th)

                LDPatch = LD[:, y1: y1 + th,x1: x1 + tw]
                FDPatch = FD[:, y1: y1 + th,x1: x1 + tw]
                w1Patch = w1[:, y1: y1 + th, x1: x1 + tw]
                w2Patch = w2[:, y1: y1 + th, x1: x1 + tw]
                #w3Patch = w3[:, y1: y1 + th, x1: x1 + tw]
                numAir = (FDPatch < 0.0000001).sum()
        return LDPatch, FDPatch, w1Patch, w2Patch

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, LD, FD, w1,w2):
        if random.random() < 0.5:
            LD = np.copy(np.fliplr(LD[0,:,:]))
            LD = LD[np.newaxis]
            FD = np.copy(np.fliplr(FD[0,:,:]))
            FD = FD[np.newaxis]
            w1 = np.copy(np.fliplr(w1[0, :, :]))
            w1 = w1[np.newaxis]
            w2 = np.copy(np.fliplr(w2[0, :, :]))
            w2 = w2[np.newaxis]
            #w3 = np.copy(np.fliplr(w3[0, :, :]))
            #w3 = w3[np.newaxis]
        return LD, FD, w1, w2


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, LD, FD,w1,w2):
        if random.random() < 0.5:
            LD = np.copy(np.flipud(LD[0,:,:]))
            LD = LD[np.newaxis]
            FD = np.copy(np.flipud(FD[0,:,:]))
            FD = FD[np.newaxis]
            w1 = np.copy(np.flipud(w1[0, :, :]))
            w1 = w1[np.newaxis]
            w2 = np.copy(np.flipud(w2[0, :, :]))
            w2 = w2[np.newaxis]
            #w3 = np.copy(np.flipud(w3[0, :, :]))
            #w3 = w3[np.newaxis]
        return LD, FD, w1, w2

class RandomRotate(object):
    """Random rotation of the image from -angle to angle (in degrees)
    This is useful for dataAugmentation, especially for geometric problems such as FlowEstimation
    angle: max angle of the rotation
    interpolation order: Default: 2 (bilinear)
    reshape: Default: false. If set to true, image size will be set to keep every pixel in the image.
    diff_angle: Default: 0. Must stay less than 10 degrees, or linear approximation of flowmap will be off.
    """

    def __init__(self, angle, diff_angle=0, order=2, reshape=False):
        self.angle = angle
        self.reshape = reshape
        self.order = order
        self.diff_angle = diff_angle

    def __call__(self, inputs,target):
        applied_angle = random.uniform(-self.angle,self.angle)
        diff = random.uniform(-self.diff_angle,self.diff_angle)
        angle1 = applied_angle - diff/2
        angle2 = applied_angle + diff/2
        angle1_rad = angle1*np.pi/180

        h, w, _ = target.shape

        def rotate_flow(i,j,k):
            return -k*(j-w/2)*(diff*np.pi/180) + (1-k)*(i-h/2)*(diff*np.pi/180)

        rotate_flow_map = np.fromfunction(rotate_flow, target.shape)
        target += rotate_flow_map

        inputs[0] = ndimage.interpolation.rotate(inputs[0], angle1, reshape=self.reshape, order=self.order)
        inputs[1] = ndimage.interpolation.rotate(inputs[1], angle2, reshape=self.reshape, order=self.order)
        target = ndimage.interpolation.rotate(target, angle1, reshape=self.reshape, order=self.order)
        # flow vectors must be rotated too! careful about Y flow which is upside down
        target_ = np.copy(target)
        target[:,:,0] = np.cos(angle1_rad)*target_[:,:,0] + np.sin(angle1_rad)*target_[:,:,1]
        target[:,:,1] = -np.sin(angle1_rad)*target_[:,:,0] + np.cos(angle1_rad)*target_[:,:,1]
        return inputs,target

class RandomRot90(object):
    def __call__(self, LD, FD,w1,w2):
        k = np.random.randint(0,4)
        LD = np.copy(np.rot90(LD, k=k, axes=(1,2)))
        FD = np.copy(np.rot90(FD, k=k, axes=(1,2)))
        w1 = np.copy(np.rot90(w1, k=k, axes=(1, 2)))
        w2 = np.copy(np.rot90(w2, k=k, axes=(1, 2)))
        #w3 = np.copy(np.rot90(w3, k=k, axes=(1, 2)))
        return LD, FD,w1,w2

class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs,target):
        h, w, _ = inputs[0].shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        inputs[0] = inputs[0][y1:y2,x1:x2]
        inputs[1] = inputs[1][y3:y4,x3:x4]
        target = target[y1:y2,x1:x2]
        target[:,:,0] += tw
        target[:,:,1] += th

        return inputs, target

class RandomColorWarp(object):
    def __init__(self, mean_range=0, std_range=0):
        self.mean_range = mean_range
        self.std_range = std_range

    def __call__(self, inputs, target):
        random_std = np.random.uniform(-self.std_range, self.std_range, 3)
        random_mean = np.random.uniform(-self.mean_range, self.mean_range, 3)
        random_order = np.random.permutation(3)

        inputs[0] *= (1 + random_std)
        inputs[0] += random_mean

        inputs[1] *= (1 + random_std)
        inputs[1] += random_mean

        inputs[0] = inputs[0][:,:,random_order]
        inputs[1] = inputs[1][:,:,random_order]

        return inputs, target



class CoNormalize(object):
    """ Exactly the normalize in torchvision.transforms
    except that I specify different mean and std for LD and FD
    the default value are from the synthetic dataset 
    """
    def __init__(self, FDMean=[270.1626], FDStd=[413.2266], LDMean=[270.8705], LDStd=[404.4441], inplace=False):
        self.LDMean = LDMean
        self.FDMean = FDMean
        self.FDStd = FDStd
        self.LDStd = LDStd
        self.inplace =False
    
    def __call__(self, LD, FD):
        LD = F.normalize(LD, self.LDMean, self.LDStd, self.inplace)
        FD = F.normalize(FD, self.FDMean, self.FDStd, self.inplace)
        return LD, FD


class ClipNormalize(object):
    def __init__(self, lower, upper):
        self.lower = torch.tensor(lower).float()
        self.upper = torch.tensor(upper).float()

    def __call__(self, LD, FD,w1,w2):
        LD = torch.min(torch.max(LD, self.lower), self.upper)
        LD = (LD - self.lower) / (self.upper - self.lower)
        FD = torch.min(torch.max(FD, self.lower), self.upper)
        FD = (FD - self.lower) / (self.upper - self.lower)
        w1 = torch.min(torch.max(w1, self.lower), self.upper)
        w1 = (w1 - self.lower) / (self.upper - self.lower)
        w2 = torch.min(torch.max(w2, self.lower), self.upper)
        w2 = (w2 - self.lower) / (self.upper - self.lower)
        #w3 = torch.min(torch.max(w3, self.lower), self.upper)
        #w3 = (w3 - self.lower) / (self.upper - self.lower)

        return LD.float(), FD.float(), w1.float(), w2.float()

class DicomStandardNormalize(object):
    """
    according to https://dicom.innolitics.com/ciods/ct-image/voi-lut/00281050
    c: (0028,1050) window center
    w: (0028,1051) window width
    """
    def __init__(self, c, w, RescaleSlope, RescaleIntercept, ymin=0.0, ymax=1.0):
        self.ymin = float(ymin)
        self.ymax = float(ymax)
        self.c = float(c)
        self.w = float(w)
        self.slope = float(RescaleSlope)
        self.intercept = float(RescaleIntercept)
        
    def __call__(self, LD, FD):

        if self.slope != 1.0:
            LD = LD * self.slope + self.intercept
            FD = FD * self.slope + self.intercept
        else:
            LD = LD + self.intercept
            FD = FD + self.intercept
        LD = ((LD - (self.c-0.5)) / (self.w-1.0) + 0.5) * (self.ymax-self.ymin) + self.ymin
        LD = torch.max(LD, torch.tensor(self.ymin).float())
        LD = torch.min(LD, torch.tensor(self.ymax).float())
        FD = ((FD - (self.c-0.5)) / (self.w-1.0) + 0.5) * (self.ymax-self.ymin) + self.ymin
        FD = torch.max(FD, torch.tensor(self.ymin).float())
        FD = torch.min(FD, torch.tensor(self.ymax).float())
        return LD.float(), FD.float()

class Clip2LungNormalize(ClipNormalize):
    """I assume that the input ct images is HU+1024, i.e. air will be 24
    Therefore, lung ([-700, -600] in HU unit) will be in this range
    """
    def __init__(self):
        super(Clip2LungNormalize, self).__init__(1024.0+40.0-700.0, 1024.0+40.0-600.0)


class NoiseReduce(object):
    def __init__(self, noiseReduceMaxLevel=0.5):
        self.noiseReduceMaxLevel = noiseReduceMaxLevel
    def __call__(self, LD, FD):
        noiseReduceLevel = np.random.uniform(low=0.0, high=self.noiseReduceMaxLevel)
        return noiseReduceLevel * FD + (1-noiseReduceLevel) * LD, FD



if __name__=='__main__':
    # LD = torch.rand([1,512,512]).float()*2000
    # FD = torch.rand([1,512,512]).float()*2000
    # transform = RandomCrop(64, 0.95)
    # print(LD.shape)
    # LD, FD = transform(LD, FD)
    # print(LD.shape)
    # import sys
    # sys.exit()

    import initialize
    opt = initialize.InitParamsYaml()
    transform = Option2Transforms(opt, train=True)
    
    trainData = np.load('data/LCTSC/train_poisson_30.npz')
    FD = trainData['FD'][0,:,:,:]
    LD = trainData['LD'][0,:,:,:]
    LDAug, FDAug = transform(LD, FD)

    from matplotlib import pyplot as plt
    plt.imshow(FD[0], cmap='gray')
    plt.show()
    plt.imshow(FDAug[0], cmap='gray')
    plt.show()
