import math
import pydicom
import numpy as np
from matplotlib import pyplot as plt
import skimage.measure as measure


def poi_normalize(ds, img=None):
    if img is None:
        img = ds.pixel_array
    img = img.astype(np.float64)
    img = img * ds.RescaleSlope + ds.RescaleIntercept
    # img= img.astype(np.int16)
    vis_lower = (ds.WindowCenter - ds.WindowWidth) / 2
    vis_upper = (ds.WindowCenter + ds.WindowWidth) / 2
    img = np.minimum(np.maximum(img, vis_lower), vis_upper)
    img = (img - vis_lower) / (vis_upper - vis_lower)
    return img

def compare_psnr(inputs, targets):
    """compute the psnr of a batch, return as a list, images are assumed in [0,1]"""
    assert inputs.shape[0] == targets.shape[0]
    psnrs=[]
    for i in range(inputs.shape[0]):
        ainput = inputs[i].astype(np.float64)
        target = targets[i].astype(np.float64)
        mse = np.mean((ainput - target)**2)
        if mse == 0:
            psnrs.append(float(100.0))
        else:
            psnrs.append(20 * math.log10(1.0 / math.sqrt(mse)))
    return psnrs

def compare_ssim(inputs, targets):
    """compute the psnr of a batch, return as a list, images are assumed in [0,1]"""
    assert inputs.shape[0] == targets.shape[0]
    ssims=[]
    for i in range(inputs.shape[0]):
        ainput = np.squeeze(inputs[i].astype(np.float64))
        target = np.squeeze(targets[i].astype(np.float64))
        ssim = measure.compare_ssim(ainput, target, data_range=1.0)
        ssims.append(ssim)
    return ssims



if __name__=='__main__':
    ds = pydicom.read_file('data/000000.dcm')
    img = poi_normalize(ds)
    plt.imshow(img, cmap='gray')
    plt.show()