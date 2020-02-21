import numpy
import torch

from pybio.torch.transformations.generic import EnsureNumpy, EnsureTorch


def test_EnsureTorch():
    trf = EnsureTorch()
    ipt_tensors = [numpy.empty(s) for s in [(1, 2), (3, 4)]]
    out_tensors = trf.apply(*ipt_tensors)
    assert all(isinstance(out, torch.Tensor) for out in out_tensors)


def test_EnsureNumpy():
    trf = EnsureNumpy()
    ipt_tensors = [torch.from_numpy(numpy.empty(s)) for s in [(1, 2), (3, 4)]]
    out_tensors = trf.apply(*ipt_tensors)
    assert all(isinstance(out, numpy.ndarray) for out in out_tensors)
