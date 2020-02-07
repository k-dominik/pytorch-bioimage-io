import torch
from pybio.torch.transformations.base import PyBioTorchTransformation


# TODO would be nice to auto-generate this
class Sigmoid(PyBioTorchTransformation):
    """ Sigmoid activation
    """

    def apply_to_chosen(self, tensor):
        return torch.sigmoid(tensor)
