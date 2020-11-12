from torch.nn.modules.module import Module
from ..functions.rbf_neigh import rbf_neighbours

'''
Generate RBF neighbourhood from scattered points
Input:
    r, c: row and column indices of shape [N, NUM_POINTS]
Output:
    p_neigh, a_neigh, dx_neigh, dy_neigh: all are [N, H, W, K] tensor, denoting
    the point indices in the range [0, NUM_POINTS), applicability masks in {0, 1},
    coordinate distances along x- and y-axis of the K nearest spatial neighbours,
    respectively.
'''
class RBFNeighbours(Module):

    def __init__(self, radius, k, h, w):
        super(RBFNeighbours, self).__init__()
        self.func = rbf_neighbours(radius, k, h, w)

    def reset_params(self):
        return

    def forward(self, r, c):
        return self.func(r, c)

    def __repr__(self):
        return self.__class__.__name__

