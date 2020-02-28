import numpy
import torch

from pybio.torch.transformations.generic import AverageChannels


def test_AverageChannels():
    in_shape = (1, 8, 16, 16)
    exp_shape = (1, 1, 16, 16)
    ipt_tensors = [torch.from_numpy(numpy.ones(in_shape))]

    channel_ranges = [(None, None), (0, 8), (1, 3), (4, 7), (3, None), (None, 5)]
    for startc, stopc in channel_ranges:
        trf = AverageChannels(startc, stopc)
        out_tensor = trf.apply(*ipt_tensors)[0]
        assert tuple(out_tensor.shape) == exp_shape
        slice_ = numpy.s_[:, startc:stopc]
        exp_tensor = torch.mean(ipt_tensors[0][slice_], 1, keepdim=True)
        assert torch.allclose(out_tensor, exp_tensor)
