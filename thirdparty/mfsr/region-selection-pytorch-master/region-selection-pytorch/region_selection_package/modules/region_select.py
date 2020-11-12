from torch.nn.modules.module import Module
from ..functions.region_select import region_selection

'''
Select a set of sub-regions from the input per-pixel maps around the given coordinates.
Input:
    x: [N, C, H, W, MH, MW], where MH and MW are the size of per-pixel maps
    indices: [N, H, W, 2] pixel coordinates (height-first)
Output:
    [N, RH, RW, H, W] tensor, where RH and RW are region size
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

