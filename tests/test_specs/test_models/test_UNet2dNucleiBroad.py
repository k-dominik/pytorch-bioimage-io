from pathlib import Path
from io import BytesIO

import torch

from pybio.spec import load_spec


def test_UNet2dNucleiBroads():
    spec_path = (
        Path(__file__).parent
        / "../../../specs/models/unet2d/nuclei_broad/UNet2dNucleiBroad.model.yaml"
    )
    loaded_spec = load_spec(spec_path.as_posix(), kwargs={})
    with BytesIO() as f:
        model = loaded_spec.train(n_iterations=1, out_file=f)
        f.seek(0)
        loaded = torch.load(f)

    state = model.state_dict()
    for t in state:
        assert t in loaded
        assert torch.equal(state[t], loaded[t])
