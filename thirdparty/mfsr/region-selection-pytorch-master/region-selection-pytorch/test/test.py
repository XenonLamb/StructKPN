import torch
import torch.nn as nn
from torch.autograd import Variable
from region_selection_package.modules.region_select import RegionSelection
from region_selection_package.functions.region_select import region_selection

from torch.autograd import gradcheck

import numpy as np

def test():
    A = Variable(torch.randn(2,6,8,5,5), requires_grad=False)
    A_ = A.cuda()

    B = torch.ones(2, 6, 8, 2) * 3
    B_ = B.cuda()

    o = RegionSelection(3, 3)(A_, B_)

    print(o.size())
    print('Module interface test passed')

if __name__=='__main__':
    test()
