from .base import IndependentTransformation


class NormalizeZeroMeanUnitVariance(IndependentTransformation):
    """ Sigmoid activation
    """
    def __init__(self, apply_to, eps=1.e-6, mean=None, std=None):
        self.eps = eps
        self.mean = mean
        self.std = std
        super().__init__(apply_to=apply_to)

    def apply_transformation(self, tensor):
        mean = tensor.mean() if self.mean is None else mean
        std = tensor.std() if self.std is None else std
        return (tensor - mean) / (std + self.eps)
