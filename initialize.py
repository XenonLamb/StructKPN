# code copied heavily from ESRGAN: https://github.com/xinntao/BasicSR 

import argparse
import logging
import os
import yaml
from collections import OrderedDict

def CreateParser():
    parser = argparse.ArgumentParser(description='Thinking fast and slow')
    parser.add_argument('experimentName', metavar='EXPERIMENTNAME', type=str,
                        help='name of the experiment')

    # Synthesized dataset generation
    parser.add_argument('--synthesizeDataset', metavar='DATASET', default=False,
                        type=str2bool, help='indicate whether to synthesize data or train network')
    parser.add_argument('--noiseLevel', default=30.0, type=float,
                        help='the level of synthesized poisson noise')

    # dataset
    parser.add_argument('--dataset', metavar='DATASET', default='LCTSC', 
                        choices=['LCTSC'], help='The dataset to be used')
    parser.add_argument('--dicomDir', default='data', type=str,
                        help='where the raw .dcm files are')
    parser.add_argument('--npyDir', default='data', type=str,
                        help='where the processed ready-to-use .npy files are')
    parser.add_argument('--discardBlackPatches', default=True, type=str2bool,
                        help='throw away all black patches')
    parser.add_argument('--saveImages', default=False, type=str2bool,
                        help='save debug images')
    parser.add_argument('--debugDir', default='images', type=str,
                        help='where the debug images are saved')
    parser.add_argument('--tensorboardDir', default='tensorboard', type=str,
                        help='where the tensorboard record are saved')                    

    # training
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--startEpoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batchSize', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--dropLast', default=True, type=str2bool,
                        help='drop the remaining data which is not enough for a whole batch')
    parser.add_argument('--lr', '--learningRate', default=1e-4, type=float,
                        metavar='LR', help='max learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weightDecay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--saveEpochs', default=1, type=int,
                        metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
    parser.add_argument('--ckptDir', default='checkpoint', type=str,
                        help='where to save model checkpoints')
    parser.add_argument('--evaluationEpochs', default=1, type=int,
                        metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
    parser.add_argument('--resume', default=False, type=str2bool,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', default='False', type=str2bool,
                        help='evaluate model on evaluation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--lossFunc', default='L1', choices=['L1','L2'],
                        help='loss function during training')

    # network architecture
    parser.add_argument('--arch', '-a', metavar='ARCH', default='kpn', choices=['kpn'], 
                        help='denoising model architecture')
    parser.add_argument('--patchSize', default=64, type=int,
                        help='patch size of training data')
    parser.add_argument('--KPNKernelSize', default=21, type=int,
                        help='KPN kernel size')
    parser.add_argument('--filterChannels', default=24, type=int,
                        help='how many filters to be used in the hidden layers')
    parser.add_argument('--numBlocks', default=24, type=int,
                        help='number of residual blocks')
    # parser.add_argument('--KPNImplementation', default=1, choices=[1,2,3,4],
    #                     help='different implementation of kpn/SeperableConvolution/WeightedAverage'
    #                     +'1: '
    #                     +'2: '
    #                     +'3: '
    #                     +'4: ')
    return parser


def InitParams():
    parser=CreateParser()
    args = parser.parse_args()

    args.dicomDir = args.dicomDir + '/' + args.dataset
    args.npyDir = args.npyDir + '/' + args.dataset
    
    # result dir
    saveDir = 'results/' + args.experimentName + '/'
    args.debugDir = saveDir + args.debugDir
    args.tensorboardDir = saveDir + args.tensorboardDir
    args.ckptDir = saveDir + args.ckptDir
    InitializeResultsFolders([args.debugDir, 
        args.tensorboardDir, args.ckptDir])
    
    # save all the arguments input in the terminal
    with open(saveDir+'log.txt', 'w') as f:
        for arg, value in sorted(vars(args).items()):
            print('Argument {}: {}'.format(arg, value))
            f.write('Argument {}: {}'.format(arg, value))
    
    return args


def InitParamsYaml():
    Loader, Dumper = OrderedYaml()
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    with open('./options/' + args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    opt['path']['logDir'] = os.path.join(
        'results', opt['name'])
    opt['path']['debugDir'] = os.path.join(
        'results', opt['name'], opt['path']['debugDir'])
    opt['path']['tensorboardDir'] = os.path.join(
        'results', opt['name'], opt['path']['tensorboardDir'])
    opt['path']['ckptDir'] = os.path.join(
        'results', opt['name'], opt['path']['ckptDir'])
    InitializeResultsFolders([
        opt['path']['logDir'],
        opt['path']['debugDir'], 
        opt['path']['tensorboardDir'], 
        opt['path']['ckptDir'],
        #opt['path']['saveDir']
    ])
    return opt

def InitParamsYaml2():
    Loader, Dumper = OrderedYaml()
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt1', type=str, help='Path to option YAML file.')
    parser.add_argument('-opt2', type=str, help='Path to option YAML file.')
    args = parser.parse_args()
    with open('./options/' + args.opt1, mode='r') as f:
        opt1 = yaml.load(f, Loader=Loader)

    opt1['path']['logDir'] = os.path.join(
        'results', opt1['name'])
    opt1['path']['debugDir'] = os.path.join(
        'results', opt1['name'], opt1['path']['debugDir'])
    opt1['path']['tensorboardDir'] = os.path.join(
        'results', opt1['name'], opt1['path']['tensorboardDir'])
    opt1['path']['ckptDir'] = os.path.join(
        'results', opt1['name'], opt1['path']['ckptDir'])
    InitializeResultsFolders([
        opt1['path']['logDir'],
        opt1['path']['debugDir'],
        opt1['path']['tensorboardDir'],
        opt1['path']['ckptDir'],
        #opt['path']['saveDir']
    ])
    with open('./options/' + args.opt2, mode='r') as f:
        opt2 = yaml.load(f, Loader=Loader)

    opt2['path']['logDir'] = os.path.join(
        'results', opt2['name'])
    opt2['path']['debugDir'] = os.path.join(
        'results', opt2['name'], opt2['path']['debugDir'])
    opt2['path']['tensorboardDir'] = os.path.join(
        'results', opt2['name'], opt2['path']['tensorboardDir'])
    opt2['path']['ckptDir'] = os.path.join(
        'results', opt2['name'], opt2['path']['ckptDir'])
    #InitializeResultsFolders([
    #    opt1['path']['logDir'],
    #    opt1['path']['debugDir'],
    #    opt1['path']['tensorboardDir'],
    #    opt1['path']['ckptDir'],
    #    #opt['path']['saveDir']
    #])
    return opt1, opt2

def InitParamsYamlDebug():
    Loader, Dumper = OrderedYaml()
    with open('./options/AAPMKPN.yml', mode='r') as f:
        opt = yaml.load(f, Loader=Loader)

    opt['path']['logDir'] = os.path.join(
        'results', opt['name'])
    opt['path']['debugDir'] = os.path.join(
        'results', opt['name'], opt['path']['debugDir'])
    opt['path']['tensorboardDir'] = os.path.join(
        'results', opt['name'], opt['path']['tensorboardDir'])
    opt['path']['ckptDir'] = os.path.join(
        'results', opt['name'], opt['path']['ckptDir'])
    InitializeResultsFolders([
        opt['path']['logDir'],
        opt['path']['debugDir'], 
        opt['path']['tensorboardDir'], 
        opt['path']['ckptDir']])
    return opt



try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def InitializeResultsFolders(dirs):
    for adir in dirs:
        if adir is not None:
            if not os.path.exists(adir):
                os.makedirs(adir)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


def setup_logger(logger_name, root, phase, level=logging.INFO, 
        screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s',
                                  datefmt='%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '.log')
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)



if __name__ == '__main__':
    # args = InitParams()
    opt = InitParamsYaml()
    print(opt)
