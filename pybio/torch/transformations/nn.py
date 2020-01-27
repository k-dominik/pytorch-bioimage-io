from typing import Sequence, Optional, List

import torch.nn

from pybio.core.transformations import ApplyToAll
from pybio.torch.transformations import CombinedTransformation


class BCELoss(CombinedTransformation):
    def __init__(self, apply_to: Optional[Sequence[int]] = None, **kwargs):
        super().__init__(apply_to=apply_to)
        assert isinstance(self.apply_to, ApplyToAll) or len(self.apply_to) == 2
        self.bce = torch.nn.BCELoss(**kwargs)

    def apply_to_chosen(self, ipt: torch.Tensor, tgt: torch.Tensor) -> List[torch.Tensor]:
        return [self.bce(ipt, tgt)]
