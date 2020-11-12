import os
import sys
import time
import logging
import numpy as np
from matplotlib import pyplot as plt
import pydicom
import torch
from tqdm import tqdm, trange
from PIL import Image
import initialize
from model import create_model
from cotransforms import Option2Transforms
import dataset
import train
import visualize
from PIL import Image
from numpngw import write_png
import cv2
import torch.nn.functional as F


def save_as_16b(_img, tgt_path):
    """
    saving a numpy array [0,1] as 16 bit grayscale image
    :param _img:
    :param tgt_path:
    :return:
    """
    _img_16b = (np.clip(_img,0.0,1.0)*65535.).astype(np.uint16)
    write_png(tgt_path,_img_16b,bitdepth=16)


def input2col(input, patch_size):
    n, c, h, w = input.shape
    padding = (patch_size-1)//2
    input_allpad = F.pad(input, (padding, padding, padding, padding), mode='reflect').contiguous()
    input_im2cl_list = []
    for i in range(patch_size):
        for j in range(patch_size):
            input_im2cl_list.append(input_allpad[:, :, i:(h + i), j:(w + j)].contiguous())
    input_cat = torch.cat(input_im2cl_list, 1)
    input_cat = input_cat.view(n, patch_size ** 2, c, h, w).contiguous()  # N, kk, Cin, H, W
    input_cat = input_cat.permute(0, 2, 3, 4, 1).contiguous().view(h*w, patch_size**2).contiguous() ## H*W, KK

    return input_cat




def main():
    #### VARIOUS SETTINGS ####
    # Load options from .yml
    opt = initialize.InitParamsYaml()
    if opt['train']['evaluation']:
        # As I will save the whole opt dict into the checkpoint
        # Why not just use that
        # when evaluate, one just make sure opt['name'] is correct
        ckpt = torch.load(os.path.join(opt['path']['ckptDir'], 
            opt['train']['ckptName']), map_location='cpu')
        opt['network'] = ckpt['opt']['network']
        opt['train']['evaluation'] = True
    print(initialize.dict2str(opt))

    # Logger
    initialize.setup_logger('base', opt['path']['logDir'], 
        'train', screen=True, tofile=True)
    initialize.setup_logger('test', opt['path']['logDir'], 
        'test', screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(initialize.dict2str(opt))

    # Tensorboard settings
    version = float(torch.__version__[0:3])
    if version >= 1.1:  # PyTorch 1.1
        from torch.utils.tensorboard import SummaryWriter
    else:
        logger.info(
            'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
        from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=opt['path']['tensorboardDir'])

    # convert to NoneDict, which returns None for missing keys
    opt = initialize.dict_to_nonedict(opt)

    # Cuda things
    logger.info('Let us use {} GPUs!'.format(int(torch.cuda.device_count())))
    torch.backends.cudnn.benckmark = True


    #### DATASET & MODEL ####
    # Dataset
    if opt['datasets']['name']=='LCTSC':
        if opt['datasets']['synthesizeDataset']:
            import noise
            noise.LCTSCDicom2Numpy(opt)
    if opt['datasets']['name']=='AAPM':
        if opt['datasets']['buildNumpyDataset']:
            import noise
            noise.AAPMDicom2Numpy(opt)
        
    npyTrain = np.load(os.path.join(opt['datasets']['npyDir'], 
        opt['datasets']['train']['fileName']))
    testpath = os.path.join(opt['datasets']['npyDir'],
                 opt['datasets']['test']['fileName'])
    print("testfile path:", testpath)
    npyTest = np.load(os.path.join(opt['datasets']['npyDir'], 
        opt['datasets']['test']['fileName']))
    print(npyTest['LD'].shape)
    logger.info('finish loading numpy dataset!')


    trainDataset = dataset.LowDoseCTDataset(npyTrain['LD'], npyTrain['FD'], 
        transforms=Option2Transforms(opt, train=True),splitname="train")
    testDataset = dataset.LowDoseCTDataset(npyTest['LD'], npyTest['FD'],
        transforms=Option2Transforms(opt, train=False),splitname="test")
    trainLoader, testLoader = dataset.CreateDataLoader(opt,
        trainDataset, testDataset)
    logger.info('finish establishing dataset!')
    
    # create model & load model
    model = create_model(opt)
    if opt['train']['evaluation'] or opt['train']['resume']:
        ckpt = torch.load(os.path.join(opt['path']['ckptDir'], 
            opt['train']['ckptName']), map_location='cpu')
        logger.info('loading the model from epoch: {}\t iteration: {}'.format(
            ckpt['epoch'], ckpt['iters']))
        logger.info('the model has psnr: {}\t ssim: {}; loss: {}'.format(
            ckpt['psnr'], ckpt['ssim'], ckpt['loss']))
        model = train.loadModel(ckpt['state_dict'], model)
        logger.info('state dict has been loaded to the model successfull')
    
    # data parallel, gpu
    model = torch.nn.DataParallel(model).cuda()

    # evaluate the model
    if opt['train']['evaluation']:
        logger.info('Evaluate the model')

        #### To check the psnr of training data

        idx = 0
        LDpsnrs = []
        psnrs = []
        ssims = []
        model.eval()
        start = time.time()
        with torch.no_grad():
            for _, (LD, FD,_, _) in enumerate(testLoader):
                print('evaluating image ', idx)
                LD = LD.cuda()
                FDRecon = model.forward(LD)
                LD = LD.detach().cpu().numpy()
                FD = FD.detach().cpu().numpy()
                FDRecon = FDRecon.detach().cpu().numpy()
                psnrs.extend(visualize.compare_psnr(FDRecon, FD))
                ssims.extend(visualize.compare_ssim(FDRecon, FD))
                LDpsnrs.extend(visualize.compare_psnr(LD, FD))

                for i in range(LD.shape[0]):
                    idx += 1
                    plt.imsave(opt['path']['debugDir'] + '/{:006d}_ld_{}.png'.format(idx, opt['name']),
                               np.squeeze(LD[i]), cmap='gray')
                    plt.imsave(opt['path']['debugDir'] + '/{:006d}_fd_{}.png'.format(idx, opt['name']),
                               np.squeeze(FD[i]), cmap='gray')
                    plt.imsave(opt['path']['debugDir'] + '/{:006d}_fdrecon_{}.png'.format(idx, opt['name']),
                               np.squeeze(FDRecon[i]), cmap='gray')
                
        logger.info('psnr: {}\t ssim: {}\t time(s): {}\t'.format(
            sum(psnrs)/len(psnrs), sum(ssims)/len(ssims), time.time()-start))

        #### for Patient L506
        print('Patient L506\t psnr: {}\t ssim: {}'.format(sum(psnrs[1011:])/211, sum(ssims[1011:])/211))
        
        sys.exit()

    start_epoch = opt['train']['start_epoch']
    iters = opt['train']['start_iter']
    best_psnr = -1

    # resume training
    if opt['train']['resume']:
        start_epoch = ckpt['epoch']
        iters = ckpt['iters']
        logger.info('resume the model training from epoch: {}\t iteration: {}'
            .format(start_epoch, iters))
        best_psnr = ckpt['psnr']

    # optimizer, scheduler, criterion
    criterion = train.SetCriterion(opt)
    optimizer = train.SetOptimizer(opt, model)
    scheduler = train.SetLRScheduler(opt, optimizer)

    # training
    if not opt['train']['no_train']:
        for epoch in range(start_epoch+1, start_epoch+opt['train']['epochs']+1):
            start = time.time()
            model.train()
            losses = []
            for _, (LD, FD,STRS,COHERS) in enumerate(tqdm(trainLoader, desc="Iteration")):
                iters += 1
                LD = LD.cuda()
                FD = FD.cuda()
                STRS = STRS.cuda()
                COHERS = COHERS.cuda()
                FDRecon = model.forward(LD)
                if 'asymmetric' in opt['train']['criterion']:
                    loss = criterion(FDRecon, FD, LD)
                else:
                    if 'triple' in opt['train']['criterion']:
                        if 'tripleasym' in opt['train']['criterion']:
                            loss = criterion(FDRecon, FD, LD, STRS, COHERS)
                        else:
                            loss = criterion(FDRecon, FD,STRS, COHERS)
                    else:
                        loss = criterion(FDRecon, FD)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                losses.append(loss)
                writer.add_scalar('train_loss/iterations', loss, iters)

            writer.add_scalar('train_loss/epochs', sum(losses)/len(losses), epoch)
            logger.info('epoch: {}\t loss: {}\t time(s): {}\t'.format(
                epoch, sum(losses)/len(losses), time.time()-start))

            if epoch % opt['train']['valFreq'] == 0:
                start = time.time()
                LDpsnrs = []
                psnrs = []
                ssims = []
                model.eval()
                with torch.no_grad():
                    for _, (LD, FD,_, _ ) in enumerate(testLoader):
                        LD = LD.cuda()
                        FDRecon = model.forward(LD)
                        LD = LD.detach().cpu().numpy()
                        FD = FD.detach().cpu().numpy()
                        FDRecon = FDRecon.detach().cpu().numpy()
                        psnrs.extend(visualize.compare_psnr(FDRecon, FD))
                        ssims.extend(visualize.compare_ssim(FDRecon, FD))
                        LDpsnrs.extend(visualize.compare_psnr(LD, FD))

                psnrAvg = sum(psnrs)/len(psnrs)
                ssimAvg = sum(ssims)/len(ssims)
                writer.add_scalar('test_psnr/epochs', psnrAvg, epoch)
                writer.add_scalar('test_ssim/epochs', ssimAvg, epoch)
                logger.info('Low-dose image psnr: {}'.format(sum(LDpsnrs)/len(LDpsnrs)))
                logger.info('epoch: {}\t iterations: {}\t psnr: {}\t ssim: {}\t time(s): {}\t'.format(
                    epoch, iters, psnrAvg, ssimAvg, time.time()-start))

                logger.info('saving the model')
                torch.save({
                    'state_dict': model.state_dict(),
                    'epoch': epoch,
                    'iters': iters,
                    'psnr': psnrAvg,
                    'ssim': ssimAvg,
                    'loss': sum(losses) / len(losses),
                    'opt': opt
                }, opt['path']['ckptDir'] + '/latest_ckpt.t7')

                if (psnrAvg > best_psnr):
                    logger.info('saving the model')
                    torch.save({
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'iters': iters,
                        'psnr': psnrAvg,
                        'ssim': ssimAvg,
                        'loss': sum(losses)/len(losses),
                        'opt': opt
                    }, opt['path']['ckptDir']+'/ckpt.t7')
                    best_psnr = psnrAvg



    if opt['train']['evaluation']:
        logger.info('Evaluate the model')

        #### To check the psnr of training data
        testTrainDataset = dataset.LowDoseCTDataset(npyTrain['LD'], npyTrain['FD'],
            transforms=Option2Transforms(opt, train=False, reduce=True))
        _, testTrainLoader = dataset.CreateDataLoader(opt, trainDataset, testTrainDataset)
        testTrainpsnrs = []
        testTrainssims = []
        testTrainInputPatches = []
        testTrainInputKernels = []
        RECORD_LIMIT = 1500
        record_count = 0
        model.eval()
        start = time.time()
        with torch.no_grad():
            for _, (LD, FD, _, _) in enumerate(testTrainLoader):
                LD = LD.cuda()

                if('KPN' in opt['network']['whichModel']):
                    FDRecon, kernels = model.forward(x=LD, verbose=True)
                    if record_count < RECORD_LIMIT:
                        record_count +=1
                        testTrainInputPatches.append(input2col(LD,patch_size=opt['network']['KPNKernelSize']).detach().cpu().numpy().astype(np.float16))
                        ksize = kernels.shape[1]
                        testTrainInputKernels.append(kernels.permute(0,2,3,1).contiguous().view(-1, ksize).contiguous().detach().cpu().numpy())

                else:
                    FDRecon = model.forward(LD)
                LD = LD.detach().cpu().numpy()
                FD = FD.detach().cpu().numpy()
                FDRecon = FDRecon.detach().cpu().numpy()
                testTrainpsnrs.extend(visualize.compare_psnr(FDRecon, FD))
                testTrainssims.extend(visualize.compare_ssim(FDRecon, FD))

        testTrainInputPatches_total = np.concatenate(testTrainInputPatches, axis=0)
        testTrainInputKernels_total = np.concatenate(testTrainInputKernels, axis=0)
        logger.info('Saving patch-kernel pairs!')
        np.savez_compressed(os.path.join(opt['path']['logDir'], 'kernel_stats_cropped_small.npz'), inputs=testTrainInputPatches_total, kernels=testTrainInputKernels_total)

        logger.info('On the training data, psnr: {}\t ssim: {}\t time(s): {}\t'.format(
            sum(testTrainpsnrs)/len(testTrainpsnrs),
            sum(testTrainssims)/len(testTrainssims), time.time()-start))

    # save test images
    logger.info('saving reconstruction results')
    idx = 0
    if opt['logger']['save_best']:
        model = create_model(opt)
        ckpt = torch.load(os.path.join(opt['path']['ckptDir'], 
            opt['train']['ckptName']), map_location='cpu')
        logger.info('loading the model from epoch: {}\t iteration: {}'.format(
            ckpt['epoch'], ckpt['iters']))
        logger.info('the model has psnr: {}\t ssim: {}; loss: {}'.format(
            ckpt['psnr'], ckpt['ssim'], ckpt['loss']))
        model = train.loadModel(ckpt['state_dict'], model)
        logger.info('state dict has been loaded to the model successfull')
        model.eval()
        model = torch.nn.DataParallel(model).cuda()
    save_grp =[888]
    if opt['network']['temperature'] is not None:
        temperature = opt['network']['temperature']
    else:
        temperature = 1.0
    LDs = []
    FDs = []
    FDRecons = []
    cpu_device = torch.device('cpu')
    model_cpu = model.module.to(cpu_device)
    for _, (LD, FD,_, _) in enumerate(testLoader):
        print('evaluating image ', idx)
        #LD = LD.cuda()
        if opt['train']['no_train']:
            FDRecon, kernels = model_cpu.forward(x=LD, verbose=True)
        else:
            FDRecon = model_cpu.forward(x=LD,temperature=temperature)
        LD = LD.detach().cpu().numpy()
        FD = FD.detach().cpu().numpy()
        FDRecon = FDRecon.detach().cpu().numpy()
        LDs.append(LD)
        FDs.append(FD)
        FDRecons.append(FDRecon)
        for i in range(LD.shape[0]):
            idx += 1
            if (idx in save_grp) or (1>0):
                cv2.imwrite(opt['path']['debugDir'] + '/{:006d}_fdrecon_{}.png'.format(idx, opt['name']), np.squeeze(FDRecon[i])*255.)
                cv2.imwrite(opt['path']['debugDir'] + '/{:006d}_fd_{}.png'.format(idx, opt['name']), np.squeeze(FD[i])*255.)
                cv2.imwrite(opt['path']['debugDir'] + '/{:006d}_ld_{}.png'.format(idx, opt['name']), np.squeeze(LD[i])*255.)
            if opt['train']['no_train'] and (idx in save_grp):
                kernels = kernels.view(kernels.shape[0], 21, 21, kernels.shape[2], kernels.shape[3]).permute(0, 3, 4, 1,
                                                                                                             2)
                (B, H, W, X, Y) = kernels.shape
                disp_kernels = np.zeros((32,32,21,21),dtype=np.float32)
                kernels = kernels.detach().cpu().contiguous().numpy()
                for id1 in range(H):
                    for id2 in range(W):
                        if (id1%16 ==1)and(id2%16==1):
                            disp_kernels[id1//16, id2//16,:, :] = np.squeeze(kernels[i][id1][id2])
                np.savez(opt['path']['debugDir'] + '/kernels888_{}.npz'.format(opt['name']), kernels = disp_kernels)

    np.savez(opt['path']['debugDir'] + '/results_{}.npz'.format(opt['name']), LD=LDs, FD=FDs, FDRecon=FDRecons)

    logger.info('finish saving all the test images')


if __name__=='__main__':
    main()