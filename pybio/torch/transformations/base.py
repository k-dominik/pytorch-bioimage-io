# TODO would be nice to have something similar to inferno's batch_function etc.
# functionality, but more flexible w.r.t implementing different dimensions.
from typing import List, Sequence, Tuple, Callable, Union

import torch

import pybio.core.transformations
from pybio.core.array import PyBioScalar


class Transformation(pybio.core.transformations.Transformation):
    pass


class IndependentTransformation(Transformation):
    """ Transformation that can be applied to all input tensors independently.
    """

    pass


class SynchronizedTransformation(Transformation):
    """ Transformation for which application to all tensors is synchronized.
    This means, some state must be known before applying it to the tensors,
    e.g. the degree before a random rotation
    """

    def set_next_state(self):
        raise NotImplementedError

    def apply(self, *tensors):
        # TODO the state might depend on some tensor properties (esp. shape)
        # inferno solves this with the 'set_random_state' and 'get_random_state' construction
        # here, we could just pass *tensors to set_next_state
        self.set_next_state()
        return super().apply(*tensors)


class Loss(pybio.core.transformations.Transformation):
    loss_callable: Callable

    def apply(
        self, *tensors: torch.Tensor, losses: List[PyBioScalar]
    ) -> Tuple[Sequence[torch.Tensor], List[PyBioScalar]]:
        losses.append(self.loss_callable(*[tensors[i] for i in self.apply_to]))
        return tensors, losses


def apply_transformations_and_losses(
    transformations: Sequence[Union[Transformation, Loss]], *tensors: torch.Tensor
) -> Tuple[List[torch.Tensor], List[PyBioScalar]]:
    """ Helper function to apply a list of transformations to input tensors.
    """
    if not all(isinstance(trafo, Transformation) or isinstance(trafo, Loss) for trafo in transformations):
        raise ValueError("Expect iterable of transformations and losses")

    losses = []
    for trafo in transformations:
        if isinstance(trafo, Transformation):
            tensors = trafo.apply(*tensors)
        elif isinstance(trafo, Loss):
            tensors, losses = trafo.apply(*tensors, losses=losses)
        else:
            raise NotImplementedError(type(trafo))

    return tensors, losses
