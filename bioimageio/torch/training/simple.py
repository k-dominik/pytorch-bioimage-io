import torch

from pathlib import Path
from torch.utils.data import DataLoader
from typing import Union, IO

from bioimageio.spec.utils import get_nn_instance

try:
    from tqdm import trange
except ImportError:
    import warnings

    warnings.warn("tqdm dependency is missing")
    trange = range


from bioimageio.spec.raw_nodes import ModelWithKwargs
from bioimageio.torch.transformations import apply_transformations


def simple_training(
    bioimageio_model: ModelWithKwargs, n_iterations: int, batch_size: int, num_workers: int, out_file: Union[str, Path, IO[bytes]]
) -> torch.nn.Module:
    """ Simplified training loop.
    """
    if isinstance(out_file, str) or isinstance(out_file, Path):
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True)

    model = get_nn_instance(bioimageio_model)

    # instantiate all training parameters from the training config
    setup = bioimageio_model.spec.training.setup

    sampler = get_nn_instance(setup.sampler)

    preprocess = [get_nn_instance(prep) for prep in setup.preprocess]
    postprocess = [get_nn_instance(post) for post in setup.postprocess]

    losses = [get_nn_instance(loss_prep) for loss_prep in setup.losses]
    optimizer = get_nn_instance(setup.optimizer, params=model.parameters())

    # build the data-loader from our sampler
    loader = DataLoader(sampler, shuffle=True, num_workers=num_workers, batch_size=batch_size)

    # run the training loop
    for ii in trange(n_iterations):
        x, y = next(iter(loader))
        optimizer.zero_grad()

        x, y = apply_transformations(preprocess, x, y)
        out = model(x)
        out, y = apply_transformations(postprocess, out, y)
        losses = apply_transformations(losses, out, y)
        ll = sum(losses)
        ll.backward()

        optimizer.step()

    # save model weights
    torch.save(model.state_dict(), out_file)
    return model
