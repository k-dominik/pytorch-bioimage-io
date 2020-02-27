from pathlib import Path
from io import BytesIO

import numpy
import torch

from pybio.core.transformations import apply_transformations
from pybio.spec import load_model, utils
from pybio.spec.utils import get_instance


def test_UNet2dNucleiBroads(cache_path):
    spec_path = (
        Path(__file__).parent / "../../../specs/models/unet2d/nuclei_broad/UNet2DNucleiBroad.model.yaml"
    ).resolve()
    assert spec_path.exists(), spec_path
    pybio_model = load_model(str(spec_path), cache_path=cache_path)
    with BytesIO() as f:
        model = utils.train(pybio_model, n_iterations=1, out_file=f)
        f.seek(0)
        loaded = torch.load(f)

    state = model.state_dict()
    for t in state:
        assert t in loaded
        assert torch.equal(state[t], loaded[t])


def test_UNet2dNucleiBroads_load_weights(cache_path):
    spec_path = (
        Path(__file__).parent / "../../../specs/models/unet2d/nuclei_broad/UNet2DNucleiBroad.model.yaml"
    ).resolve()
    assert spec_path.exists(), spec_path
    pybio_model = load_model(str(spec_path), cache_path=cache_path)
    assert isinstance(pybio_model.spec.prediction.weights.source, BytesIO)


def test_UNet2dNucleiBroads_forward(cache_path):
    spec_path = (
        Path(__file__).parent / "../../../specs/models/unet2d/nuclei_broad/UNet2DNucleiBroad.model.yaml"
    ).resolve()
    assert spec_path.exists(), spec_path
    pybio_model = load_model(str(spec_path), cache_path=cache_path)
    assert isinstance(pybio_model.spec.prediction.weights.source, BytesIO)
    assert pybio_model.spec.test_input is not None
    assert pybio_model.spec.test_input.suffix  == ".npy"
    assert pybio_model.spec.test_output is not None
    assert pybio_model.spec.test_output.suffix  == ".npy"

    model: torch.nn.Module = get_instance(pybio_model)
    model_weights = torch.load(pybio_model.spec.prediction.weights.source, map_location=torch.device("cpu"))
    model.load_state_dict(model_weights)
    pre_transformations = [get_instance(trf) for trf in pybio_model.spec.prediction.preprocess]
    post_transformations = [get_instance(trf) for trf in pybio_model.spec.prediction.postprocess]

    test_ipt = numpy.load(str(pybio_model.spec.test_input))

    test_out = numpy.load(str(pybio_model.spec.test_output))

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
