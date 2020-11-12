from torch.nn.modules.module import Module
from ..functions.tile_match import tile_matching

'''
Tile-based matching module.
Input:
    src, dst: the source and target image
    offset: [N, TH, TW, 2] tensor, where TH, TW are the number of tiles along each dimension
Output:
    A [N, TH, TW, S^2] tensor of matching distances, where S is the size of the search region.
'''
class TileMatching(Module):

    def __init__(self, tile_size=16, search_radius=8):
        super(TileMatching, self).__init__()
        self.tile_size = tile_size
        self.search_radius = search_radius
        self.func = tile_matching(tile_size, search_radius)

    def reset_params(self):
        return

    def forward(self, src, dst, offset):
        return self.func(src, dst, offset)

    def __repr__(self):
        return self.__class__.__name__

