import torch

from pybio.torch.transformations.base import IndependentTransformation


class AsType(IndependentTransformation):
    def __init__(self, dtype: str, non_blocking: bool, **super_kwargs):
        super().__init__(**super_kwargs)
        torch_dtype = getattr(torch, dtype)
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"torch.{dtype}")

        self.kwargs = {"dtype": torch_dtype, "non_blocking": non_blocking}

    def apply_to_one(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.type(**self.kwargs)
