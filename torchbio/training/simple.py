from tqdm import trange
import torch
from torch.utils.data import DataLoader
from torchbio.transformations import apply_transformations


# TODO config is just a stub object right now, adapt this to
# the actual config object from pythonbioimageio
def simple_training(config, n_iterations=500, batch_size=4, num_workers=2, out_file="./weights.pytorch"):
    """ Simplified training loop.
    """

    # instantiate the model from the model config
    model_class = config.object_
    model_kwargs = config.kwargs
    model = model_class(**model_kwargs)

    # instaniate all training parameters from the training config
    train_config = config.training.setup

    reader_class = train_config.reader.object_
    reader_kwargs = train_config.reader.kwargs
    reader = reader_class(**reader_kwargs)

    sampler_class = train_config.sampler.object_
    sampler_kwargs = train_config.sampler.kwargs
    sampler = sampler_class(reader, **sampler_kwargs)

    preprocess_config = train_config.preprocess
    preprocess = [conf.object_(**conf.kwargs) for conf in preprocess_config]

    loss_config = train_config.loss
    loss = [conf.object_(**conf.kwargs) for conf in loss_config]

    optimizer_class = train_config.optimizer.object_
    optimizer_kwargs = train_config.optimizer.kwargs
    optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)

    # build the data-loader from our sampler
    loader = DataLoader(sampler, shuffle=True, num_workers=num_workers, batch_size=batch_size)

    # run the training loop
    for ii in trange(n_iterations):
        x, y = next(iter(loader))
        optimizer.zero_grad()

        x, y = apply_transformations(preprocess, x, y)
        out = model(x)

        # we assume that the actual loss function is the last entry in
        # the list of loss transformations
        out, y = apply_transformations(loss[:-1], out, y)
        ll = loss[-1](out, y)

        ll.backward()
        optimizer.step()

    # save model weights
    torch.save(model.state_dict(), out_file)
