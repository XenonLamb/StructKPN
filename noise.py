# Add poisson noise to the reconstructed projection, 
# then transform it back to image
import os
import time
import astra
import pydicom
import glob
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from visualize import *

#### I think I will normalize while establishing the dataset
# from cotransforms import DicomStandardNormalize ## That one use pytorch api
def DicomStandardNormalizeNumpy(img, c, w, RescaleSlope, RescaleIntercept, 
    ymin=0.0, ymax=1.0):
    """
    according to https://dicom.innolitics.com/ciods/ct-image/voi-lut/00281050
    c: (0028,1050) window center
    w: (0028,1051) window width
    """
    img = img.astype(float)
    ymin = float(ymin)
    ymax = float(ymax)
    c = float(c)
    w = float(w)
    slope = float(RescaleSlope)
    intercept = float(RescaleIntercept)
    
    # to HU unit
    if RescaleSlope != 1.0:
        img = img * RescaleSlope + RescaleIntercept
    else:
        img = img + RescaleIntercept

    # Clip according to window length and window center
    img = ((img - (c-0.5)) / (w-1.0) + 0.5) * (ymax-ymin) + ymin
    img = np.minimum(np.maximum(img, ymin), ymax)

    return img.astype(float)

def CTPoisson(ds, fanbeam=False, noisy_level=1.0):
    img = ds.pixel_array.astype(np.float64) / noisy_level
    # img = scipy.io.loadmat('data/phantom.mat')['phantom256'] * 400.0
    # ds.Rows = 256
    # ds.Columns = 256
    # define the geometry
    vol_geom = astra.create_vol_geom(ds.Rows,ds.Columns)
    num_detectors = 1024
    angles = np.linspace(0, np.pi, 180)
    if fanbeam:
        proj_geom = astra.create_proj_geom(
            'fanflat',
            float(ds.get_item([0x0018, 0x0090]).value) / num_detectors,
            num_detectors,
            angles,
            float(ds.get_item([0x0018, 0x1111]).value),
            float(ds.get_item([0x0018, 0x1110]).value) - float(ds.get_item([0x0018, 0x1111]).value),
        )
    else:
        proj_geom = astra.create_proj_geom(
            'parallel', 
            float(ds.get_item([0x0018, 0x0090]).value) / num_detectors,
            num_detectors, 
            angles
        )

    # img back to projection
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    sino_id, sino = astra.create_sino(img, proj_id)

    # reconstruction
    recon_id = astra.data2d.create('-vol', vol_geom)
    cfg = astra.astra_dict('SIRT_CUDA')
    cfg['ReconstructionDataId'] = recon_id
    cfg['ProjectionDataId'] = sino_id
    # cfg['ProjectionId'] = proj_id
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, 150)
    recon = astra.data2d.get(recon_id)
    
    # noise up projection data
    noisy_sino = np.random.poisson(sino)
    noisy_sino_id = astra.data2d.create('-sino', proj_geom, noisy_sino)

    # reconstuction noisy
    recon2_id = astra.data2d.create('-vol', vol_geom)
    cfg2 = astra.astra_dict('SIRT_CUDA')
    cfg2['ReconstructionDataId'] = recon2_id
    cfg2['ProjectionDataId'] = noisy_sino_id
    alg2_id = astra.algorithm.create(cfg2)
    astra.algorithm.run(alg2_id, 150)
    recon2 = astra.data2d.get(recon2_id)

    astra.algorithm.delete(alg_id)
    astra.algorithm.delete(alg2_id)
    astra.data2d.delete(proj_id)
    astra.data2d.delete(sino_id)
    astra.data2d.delete(recon_id)
    astra.data2d.delete(noisy_sino_id)
    astra.data2d.delete(recon2_id)
    astra.projector.delete(proj_id)

    img = img * noisy_level
    recon = recon * noisy_level
    recon2 = recon2 * noisy_level

    return img, sino, recon, noisy_sino, recon2

    # import pylab
    # pylab.gray()
    # pylab.figure(1)
    # pylab.imshow(img)
    # pylab.figure(2)
    # pylab.imshow(sino)
    # pylab.figure(3)
    # pylab.imshow(recon)
    # pylab.show()
    

def LCTSCDicomList(dicomDir, phase):
    if phase == 'train':
        phaseDir = 'LCTSC-Train-S1*' 
    if phase == 'test':
        phaseDir = 'LCTSC-Test-S1*' 
    dsList = glob.glob(os.path.join(dicomDir, phaseDir,'*','*','*.dcm'))
    return dsList

def LCTSCDicom2Numpy(opt):
    for phase in ['train','test']:
        FDs = []
        LDs = []
        dsList = LCTSCDicomList(opt['datasets']['dicomDir'], phase)
        numImages = len(dsList)
        for i, dsFile in enumerate(dsList):
            # print('{}: {}'.format(i, dsFile))
            ds = pydicom.read_file(dsFile)
            if not hasattr(ds,'pixel_array'):
                continue
            if opt['datasets']['syntheticSettings']['noiseName'] == 'Poisson':
                FD, _, _, _, LD = CTPoisson(ds, fanbeam=False, 
                    noisy_level=opt['datasets']['syntheticSettings']['noiseLevel'])
                # 512*512 -> 1*512*512
                FD = np.expand_dims(FD, axis=0)
                LD = np.expand_dims(LD, axis=0)
            FDs.append(FD)
            LDs.append(LD)
            if i%10 == 0:
                print('{} out of {} has been proceed for {} data'.format(i, numImages, phase))
        # savez_compressed, then pretty slow?
        np.savez(
        # np.savez_compressed( 
            os.path.join(opt['datasets']['npyDir'], 
            opt['datasets'][phase]['fileName']), 
            FD=np.array(FDs), LD=np.array(LDs))

def AAPMDicomList(dicomDir, folders, dataFormat='3mm'):
    assert dataFormat in ['1mm', '3mm', 'proj']
    if dataFormat == '1mm':
        FDFolder, LDFolder = 'full_1mm', 'quarter_1mm'
    elif dataFormat == '3mm':
        FDFolder, LDFolder = 'full_3mm', 'quarter_3mm'
    elif dataFormat == 'proj':
        FDFolder, LDFolder = 'full_DICOM-CT-PD', 'quarter_DICOM-CT-PD'

    FDList = []
    LDList = []
    for folder in folders:
        if dataFormat != 'proj':
            FDDir = os.path.join(dicomDir, folder, FDFolder, '*.IMA')
            LDDir = os.path.join(dicomDir, folder, LDFolder, '*.IMA')
        else:
            raise NotImplementedError
        FDList.extend(sorted(glob.glob(FDDir)))
        LDList.extend(sorted(glob.glob(LDDir))) 
        assert len(FDList) == len(LDList) # just to make sure everything is correct 
    
    return FDList, LDList

def AAPMDicom2Numpy(opt):
    for group in ['group1', 'group2']:
        FDList, LDList = AAPMDicomList(
            opt['datasets']['dicomDir'],
            opt['datasets'][group],
            dataFormat=opt['datasets']['format']
        )
        FDs = []
        LDs = []
        numImages = len(FDList)
        for i, (FDFile, LDFile) in enumerate(zip(FDList, LDList)):
            FDds = pydicom.read_file(FDFile)
            LDds = pydicom.read_file(LDFile)
            FDimg = FDds.pixel_array
            LDimg = LDds.pixel_array

            ## debug
            assert FDds.RescaleSlope == LDds.RescaleSlope
            assert FDds.RescaleIntercept == LDds.RescaleIntercept
            assert FDds.WindowCenter == LDds.WindowCenter
            assert FDds.WindowWidth == LDds.WindowWidth
            ## debug

            if len(FDds.WindowWidth) > 1:
                c = FDds.WindowCenter[0]
                w = FDds.WindowWidth[0]
            else:
                c = FDds.WindowCenter
                w = FDds.WindowWidth
            RescaleSlope = FDds.RescaleSlope
            RescaleIntercept = FDds.RescaleIntercept

            LDimg = DicomStandardNormalizeNumpy(
                LDimg, c, w, RescaleSlope, RescaleIntercept)
            FDimg = DicomStandardNormalizeNumpy(
                FDimg, c, w, RescaleSlope, RescaleIntercept)

            FDimg = np.expand_dims(FDimg, axis=0)
            LDimg = np.expand_dims(LDimg, axis=0)
            
            FDs.append(FDimg)
            LDs.append(LDimg)
            if i%10 == 0:
                print('{} out of {} has been proceed for {}'.format(
                    i, numImages, group))
        
        FDs=np.array(FDs)
        LDs=np.array(LDs)
        np.savez(
        # np.savez_compressed( 
            os.path.join(opt['datasets']['npyDir'], 
            opt['datasets'][group+'Name']), 
            FD=FDs, LD=LDs)
    
#### some details are unclear
# def AAPMDicom2NumpyCPCE(opt):
#     for group in ['group1', 'group2']:
#         FDList, LDList = AAPMDicomList(
#             opt['datasets']['dicomDir'],
#             opt['datasets'][group],
#             dataFormat=opt['datasets']['format']
#         )
#         FDs = []
#         LDs = []
#         LDpsnr = []
#         numImages = len(FDList)
#         for i, (FDFile, LDFile) in enumerate(zip(FDList, LDList)):
#             FDds = pydicom.read_file(FDFile)
#             LDds = pydicom.read_file(LDFile)
#             FDimg = FDds.pixel_array
#             LDimg = LDds.pixel_array

#             #### To check how they calculate psnr/ssim


if __name__=='__main__':

    # build a dataset
    import initialize
    opt = initialize.InitParamsYaml()

    # AAPMDicom2Numpy(opt)
    npyTrain = np.load(os.path.join(opt['datasets']['npyDir'], 
        opt['datasets']['train']['fileName']))
    npyTest = np.load(os.path.join(opt['datasets']['npyDir'], 
        opt['datasets']['test']['fileName']))
    LDs = npyTest['LD']
    FDs = npyTest['FD']

    print(LDs.shape)
    print(FDs.shape)

    from matplotlib import pyplot as plt
    # plt.imshow(FDs[0,0], cmap='gray')
    # plt.show()
    # plt.imshow(LDs[0,0], cmap='gray')
    # plt.show()
    for i in range(LDs.shape[0]):
        plt.imsave('', np.squeeze(LDs[i,0,:,:]))

    """For LCTSC synthetic data
    start = time.time()
    LCTSCDicom2Numpy(opt)
    print('Time for building up the numpy dataset: {}'.format(time.time()-start))
    """ 

    #### check poisson noise and reconstruction
    # import pylab
    # pylab.gray()
    # file_list = sorted(glob.glob('data/LCTSC/LCTSC-Train-S1-001/*/0*/*'))
    # for file_name in file_list:
    #     ds = pydicom.read_file(file_name)
    #     img = ds.pixel_array
    #     pylab.figure()
    #     pylab.imshow(img)
    #     pylab.title(file_name[-7:])
    #     pylab.show()

    # ds = pydicom.read_file('data/000000.dcm')
    # # img = ds.pixel_array
    # # img = scipy.io.loadmat('data/phantom.mat')['phantom256']
    # img, sino, recon, noisy_sino, recon2 = CTPoisson(ds, fanbeam=False, noisy_level=30)
    # print(type(img[0,0]))
    # print(img)
    # print(recon2)

    # # plt.subplot(2,2,1)
    # # plt.imshow(img, cmap='gray')
    # # plt.title('original ct scan')
    # # plt.subplot(2,2,2)
    # # plt.imshow(sino, cmap='gray')
    # # plt.title('projection data')
    # # plt.subplot(2,2,3)
    # # plt.imshow(recon, cmap='gray')
    # # plt.title('reconstructed ct from projection')
    # # plt.subplot(2,2,4)
    # # plt.imshow(recon2, cmap='gray')
    # # plt.title('reconstructed ct from noisy projection')

    # img = poi_normalize(ds, img=img)
    # recon = poi_normalize(ds, img=recon)
    # recon2 = poi_normalize(ds, img=recon2)

    # plt.imshow(img, cmap='gray')
    # plt.title('original ct scan')
    # plt.show()
    # plt.imshow(sino, cmap='gray')
    # plt.title('projection')
    # plt.show()
    # plt.imshow(recon, cmap='gray')
    # plt.title('reconstructed ct from projection')
    # plt.show()
    # plt.imshow(recon2, cmap='gray')
    # plt.title('reconstructed ct from noisy projection')
    # plt.show()

    
    # np.save('', img)
