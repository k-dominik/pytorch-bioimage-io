from pathlib import Path
from io import BytesIO

import numpy
import torch

# from pybio.core.transformations import apply_transformations
from pybio.spec import load_spec, load_model_spec, utils
from pybio.spec.raw_nodes import URI
from pybio.spec.utils import get_instance


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
    model_spec = load_spec(spec_path)
    assert isinstance(model_spec.weights["pytorch_state_dict"].source, URI)


def test_UNet2dNucleiBroads_forward():
    spec_path = (
        Path(__file__).parent / "../../../specs/models/unet2d_nuclei_broad/UNet2DNucleiBroad.model.yaml"
    ).resolve()
    assert spec_path.exists(), spec_path
    model_spec = load_spec(spec_path)
    assert isinstance(model_spec.prediction.weights.source, BytesIO)
    assert model_spec.test_input is not None
    assert model_spec.test_input.suffix  == ".npy"
    assert model_spec.test_output is not None
    assert model_spec.test_output.suffix  == ".npy"

    model: torch.nn.Module = get_instance(model_spec)
    model_weights = torch.load(model_spec.prediction.weights.source, map_location=torch.device("cpu"))
    model.load_state_dict(model_weights)
    pre_transformations = [get_instance(trf) for trf in model_spec.prediction.preprocess]
    post_transformations = [get_instance(trf) for trf in model_spec.prediction.postprocess]

    test_ipt = numpy.load(str(model_spec.test_input))

    test_out = numpy.load(str(model_spec.test_output))

    assert hasattr(model, "forward")
    preprocessed_inputs = apply_transformations(pre_transformations, test_ipt)
    assert isinstance(preprocessed_inputs, list)
    assert len(preprocessed_inputs) == 1
    preprocessed_inputs = apply_transformations(pre_transformations, test_ipt)
    out = model.forward(preprocessed_inputs[0])
    postprocessed_outputs = apply_transformations(post_transformations, out)
    assert isinstance(postprocessed_outputs, list)
    assert len(postprocessed_outputs) == 1
    out = postprocessed_outputs[0]
    assert numpy.allclose(test_out, out)
