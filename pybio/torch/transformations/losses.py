from typing import Sequence, Optional

import torch.nn

from pybio.torch.transformations.base import Loss


class BCELoss(Loss):
    def __init__(self, apply_to: Optional[Sequence[int]] = None, **kwargs):
        super().__init__(apply_to=apply_to)
        assert len(self.apply_to) == 2, self.apply_to
        self.loss_callable = torch.nn.BCELoss(**kwargs)
