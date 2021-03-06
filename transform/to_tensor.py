import torch

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        source_image = sample["image"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        source_image = source_image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(source_image)}