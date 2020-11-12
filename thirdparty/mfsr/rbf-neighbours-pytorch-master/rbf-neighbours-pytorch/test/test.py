import torch
import torch.nn as nn
from torch.autograd import Variable
from rbf_neighbours_package.modules.rbf_neigh import RBFNeighbours
from rbf_neighbours_package.functions.rbf_neigh import rbf_neighbours

from torch.autograd import gradcheck

import numpy as np

def test():

    r = Variable(torch.randn(2, 10).float(), requires_grad=False).cuda()
    c = Variable(torch.randn(2, 10).float(), requires_grad=False).cuda()
    r = r / r.max() * 8
    c = c / c.max() * 8

    p_neigh, a_neigh, dx_neigh, dy_neigh = RBFNeighbours(2, 9, 8, 8)(r, c)

    print(p_neigh.size())
    print('Module interface test passed')
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

if __name__=='__main__':
    test()
