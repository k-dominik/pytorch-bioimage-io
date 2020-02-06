from pathlib import Path
from io import BytesIO

import torch

from pybio.spec import load_model, utils


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
