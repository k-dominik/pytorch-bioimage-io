from typing import Union

import numpy
import torch

from pybio.torch.transformations import PyBioTorchTransformation


class AsType(PyBioTorchTransformation):
    def __init__(self, dtype: str, non_blocking: bool, **super_kwargs):
        super().__init__(**super_kwargs)
        torch_dtype = getattr(torch, dtype)
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"torch.{dtype}")

        self.kwargs = {"dtype": torch_dtype, "non_blocking": non_blocking}

    def apply_to_chosen(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.type(**self.kwargs)


class EnsureTorch(PyBioTorchTransformation):
    """
    Converts `numpy.ndarray` to `torch.Tensor`, pipes `torch.Tensor` through
    """

    def apply_to_chosen(self, tensor: Union[numpy.ndarray, torch.Tensor]):
        if isinstance(tensor, numpy.ndarray):
            if tensor.dtype == numpy.uint16:  # not supported by pytorch
                tensor = tensor.astype(numpy.int32)

            return torch.from_numpy(tensor)
        else:
            return tensor


class EnsureNumpy(PyBioTorchTransformation):
    """
    Converts `torch.Tensor` to `numpy.ndarray`, pipes `numpy.ndarray` through
    """

    def apply_to_chosen(self, tensor: Union[numpy.ndarray, torch.Tensor]):
        if not isinstance(tensor, numpy.ndarray):
            return tensor.detach().cpu().numpy()
        else:
            return tensor
