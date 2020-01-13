from pathlib import Path
from typing import Union, IO

try:
    from tqdm import trange
except ImportError:
    import warnings
    warnings.warn("tqdm dependency is missing")
    trange = range

import torch

from pybio.core.transformations import apply_transformations_and_losses
from pybio.spec.spec_types import ModelSpec

from torch.utils.data import DataLoader
from pybio.torch.transformations import apply_transformations

def simple_training(model_spec: ModelSpec, n_iterations: int, batch_size: int, num_workers: int, out_file: Union[str, Path, IO[bytes]]) -> torch.nn.Module:
    """ Simplified training loop.
    """
    if isinstance(out_file, str) or isinstance(out_file, Path):
        out_file = Path(out_file)
        out_file.parent.mkdir(exist_ok=True)

    model = model_spec.get_instance()

    # instantiate all training parameters from the training config
    train_config = model_spec.spec.training.setup

    reader = train_config.reader.get_instance()
    sampler = train_config.sampler.get_instance(reader=reader)

    preprocess = [prep.get_instance() for prep in train_config.preprocess]

    loss = [loss_prep.get_instance() for loss_prep in train_config.loss]
    optimizer = train_config.optimizer.get_instance(model.parameters())

    # build the data-loader from our sampler
    loader = DataLoader(sampler, shuffle=True, num_workers=num_workers, batch_size=batch_size)

    # run the training loop
    for ii in trange(n_iterations):
        x, y = next(iter(loader))
        optimizer.zero_grad()

        x, y = apply_transformations(preprocess, x, y)
        out = model(x)

        tensors, losses = apply_transformations_and_losses(loss, out, y)
        ll = sum(losses)
        ll.backward()

        optimizer.step()

    # save model weights
    torch.save(model.state_dict(), out_file)
    return model
