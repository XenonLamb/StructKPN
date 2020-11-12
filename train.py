import time
import inspect
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from loss import *

def SetLRScheduler(opt, optimizer):
    schedulerList=[
        'epochReciprocal', 'StepLR', 'MultiStepLR',
        'ExponentialLR', 'CosineAnnealingLR', 
        'ReduceLROnPlateau', 'CyclicLR'
    ]
    assert opt['train']['lr_scheme'] in schedulerList
    schedulerParams = opt['train']['lr_scheme_params']
    if opt['train']['lr_scheme'] == 'epochReciprocal':
        reciprocal = lambda epoch: opt['train']['lr'] / (epoch+1)
        scheduler = lrs.LambdaLR(optimizer, reciprocal)
    else:
        scheduler = dict(inspect.getmembers(lrs))[opt['train']['lr_scheme']](
            optimizer, **schedulerParams)
    return scheduler

def SetOptimizer(opt, model):
    if opt['train']['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr = opt['train']['lr'],
            betas=(opt['train']['beta1'], opt['train']['beta2']),
            weight_decay = opt['train']['weight_decay']
        )
    elif opt['train']['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr = opt['train']['lr'],
            momentum = opt['train']['momentum'],
            weight_decay = opt['train']['weight_decay'],
            dampening = opt['train']['dampening'],
            nesterov = opt['train']['nesterov']
        )
    else:
        raise NotImplementedError
    return optimizer

def SetCriterion(opt):
    if opt['train']['criterion'] == 'l2':
        criterion = nn.MSELoss()
    elif opt['train']['criterion'] == 'l1':
        criterion = nn.L1Loss()
    elif opt['train']['criterion'] == 'sobel':
        criterion = grad_sobel_loss
    elif opt['train']['criterion'] == 'l1l2':
        criterion = L1L2Loss(**opt['train']['loss_param'])
    elif opt['train']['criterion'] == 'SMAPE':
        criterion = SMAPE()
    elif opt['train']['criterion'] == 'l1gl2i':
        criterion = L1GradL2Intensity(**opt['train']['loss_param'])
    elif opt['train']['criterion'] == 'asymmetricl1':
        criterion = AsymmetricL1(**opt['train']['loss_param'])
    elif opt['train']['criterion'] == 'asymmetricl2':
        criterion = AsymmetricL2(**opt['train']['loss_param'])
    elif opt['train']['criterion'] == 'asymmetricl1l2':
        criterion = AsymmetricL1L2(**opt['train']['loss_param'])
    elif opt['train']['criterion'] == 'asymmetricSMAPE':
        criterion = AsymmetricSMAPE(**opt['train']['loss_param'])
    elif opt['train']['criterion'] == 'asymmetricdownsamplel1':
        criterion = AsymmetricDownsampleL1(**opt['train']['loss_param'])
    elif opt['train']['criterion'] == 'hardtriple':
        criterion = HardTripleLoss(**opt['train']['loss_param'])
    elif opt['train']['criterion'] == 'triple':
        criterion = WeightedTripleLoss(**opt['train']['loss_param'])
    elif opt['train']['criterion'] == 'tripleasym':
        criterion = TripleAsymLoss(**opt['train']['loss_param'])
    else:
        raise NotImplementedError('the loss function is not implemented') 
    return criterion


def loadModel(stateDict, model):
    if 'module' in list(stateDict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in stateDict.items():
            name = k[7:] # remove 'module.'
            new_state_dict[name]=v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(stateDict)
    return model

