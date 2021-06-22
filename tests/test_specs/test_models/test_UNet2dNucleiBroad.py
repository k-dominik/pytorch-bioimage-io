from pathlib import Path
from io import BytesIO

import numpy
import torch

from bioimageio.spec import load_model, load_model_spec, utils
from bioimageio.spec.raw_nodes import URI
from bioimageio.spec.utils import get_nn_instance


# def test_UNet2dNucleiBroads():
#     spec_path = (
#         Path(__file__).parent / "../../../specs/models/unet2d_nuclei_broad/UNet2DNucleiBroad.model.yaml"
#     ).resolve()
#     assert spec_path.exists(), spec_path
#     model_spec = load_spec(spec_path)
#     with BytesIO() as f:
#         model = utils.train(model_spec, n_iterations=1, out_file=f)
#         f.seek(0)
#         loaded = torch.load(f)
#
#     state = model.state_dict()
#     for t in state:
#         assert t in loaded
#         assert torch.equal(state[t], loaded[t])


def test_UNet2dNucleiBroads_load_weights():
    spec_path = (
        Path(__file__).parent / "../../../specs/models/unet2d_nuclei_broad/UNet2DNucleiBroad.model.yaml"
    ).resolve()
    assert spec_path.exists(), spec_path
    model_spec = load_model(spec_path)
    assert isinstance(model_spec.weights["pytorch_state_dict"].source, Path)
