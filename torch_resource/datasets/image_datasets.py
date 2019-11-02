import torch
import torch.utils.data


class GrayscaleImageDataset(torch.utils.data.Dataset):
    """ A dataset of grayscale images.
    We assume a 3d datasource and slice the images from the first axis
    """
    def __init__(self, data_source):
        shapes = data_source.shape
        if not all(len(shape) == 3 for shape in shapes):
            raise ValueError("Invalid data source dimenions!")
        self.data_source = data_source

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        tensors = self.data_source[index]
        return tuple(torch.from_numpy(tensor[None]) for tensor in tensors)
