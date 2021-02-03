from pathlib import Path
from typing import Union

import numpy as np
import torch

from pybio.spec.utils.transformers import load_and_resolve_spec
from pybio.spec.utils import get_instance


def convert_weights_to_torchscript(
    model_yaml: Union[str, Path],
    output_path: Union[str, Path],
    use_tracing: bool = True
):
    """ Convert model weights from format 'pytorch_state_dict' to 'torchscript'.
    """
    spec = load_and_resolve_spec(model_yaml)

    with torch.no_grad():
        # load input and expected output data
        input_data = np.load(spec.test_inputs[0]).astype('float32')
        input_data = torch.from_numpy(input_data)
        expected_output_data = np.load(spec.test_outputs[0]).astype(np.float32)

        # instantiate and trace the model
        model = get_instance(spec)
        if use_tracing:
            scripted_model = torch.jit.trace(model, input_data)
        else:
            scripted_model = torch.jit.script(model)

        # check the scripted model
        output_data = scripted_model(input_data).numpy()
        if not np.allclose(expected_output_data, output_data):
            raise RuntimeError

    # save the torchscript model
    scripted_model.save(output_path)


# TODO expose this as CLI
if __name__ == '__main__':
    path = '/home/pape/Work/bioimageio/pytorch-bioimage-io/specs/models/unet2d_nuclei_broad/UNet2DNucleiBroad.model.yaml'
    out_path = './test.pt'
    convert_weights_to_torchscript(path, out_path)
