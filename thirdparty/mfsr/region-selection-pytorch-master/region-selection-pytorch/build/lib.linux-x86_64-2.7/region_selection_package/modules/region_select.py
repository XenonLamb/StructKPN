from torch.nn.modules.module import Module
from ..functions.region_select import region_selection

'''
Select a set of sub-regions around the given coordinates.
Input:
    x: [N, C, H, W] image tensor
    indices: [N, H, W, 2] pixel coordinates (height-first)
Output:
    [N, RH, RW, H, W] tensor, where RH nad RW are region size
'''
class RegionSelection(Module):

    def __init__(self, h, w):
        super(RegionSelection, self).__init__()
        self.func = region_selection(h, w)

    def reset_params(self):
        return

    def forward(self, x, indices):
        return self.func(x, indices)

    def __repr__(self):
        return self.__class__.__name__

