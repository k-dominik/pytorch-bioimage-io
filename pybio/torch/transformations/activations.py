import torch
from pybio.torch.transformations.base import Transformation


# TODO would be nice to auto-generate this
class Sigmoid(Transformation):
    """ Sigmoid activation
    """

    def apply_to_chosen(self, tensor):
        return torch.sigmoid(tensor)
