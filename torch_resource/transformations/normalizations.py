from .base import IndependentTransformation


class NormalizeZeroMeanUnitVariance(IndependentTransformation):
    """ Sigmoid activation
    """
    def __init__(self, apply_to, eps=1.e-6):
        self.eps = eps
        super().__init__(apply_to=apply_to)

    def apply_transformation(self, tensor):
        return (tensor - tensor.mean()) / (tensor.std() + self.eps)
