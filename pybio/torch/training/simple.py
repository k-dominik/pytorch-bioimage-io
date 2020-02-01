import torch

from pathlib import Path
from torch.utils.data import DataLoader
from typing import Union, IO

from pybio.spec.utils import get_instance

try:
    from tqdm import trange
except ImportError:
    import warnings

    warnings.warn("tqdm dependency is missing")
    trange = range


from pybio.spec.node import Model
from pybio.torch.transformations import apply_transformations


def simple_training(
    pybio_model: Model, n_iterations: int, batch_size: int, num_workers: int, out_file: Union[str, Path, IO[bytes]]
) -> torch.nn.Module:
    """ Simplified training loop.
    """
    if isinstance(out_file, str) or isinstance(out_file, Path):
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True)

    model = get_instance(pybio_model)

    # instantiate all training parameters from the training config
    setup = pybio_model.spec.training.setup

    reader = get_instance(setup.reader)
    sampler = get_instance(setup.sampler, reader=reader)

    preprocess = [get_instance(prep) for prep in setup.preprocess]
    postprocess = [get_instance(post) for post in setup.postprocess]

    losses = [get_instance(loss_prep) for loss_prep in setup.losses]
    optimizer = get_instance(setup.optimizer, params=model.parameters())

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
