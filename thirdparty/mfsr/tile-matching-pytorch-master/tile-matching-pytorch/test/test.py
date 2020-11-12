import torch
import torch.nn as nn
from torch.autograd import Variable
from tile_matching_package.modules.tile_match import TileMatching
from tile_matching_package.functions.tile_match import tile_matching

from torch.autograd import gradcheck

import numpy as np

def test():
    A = Variable(torch.randn(2,3,8,8), requires_grad=True)
    A_ = A.cuda()
    B = Variable(torch.randn(2,3,8,8), requires_grad=True)
    B_ = B.cuda()

    offset = torch.zeros((2, 4, 4, 2), dtype=torch.float32)
    offset[:, 0, 0, :] = 2
    offset = offset.cuda()

    o = tile_matching(2, 2)(A_, B_, offset)
    print(o.size())
    print('Module interface test passed')

if __name__=='__main__':
    test()
