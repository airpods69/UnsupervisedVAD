import random
import math
import numbers
import collections
import numpy as np
import torch

from PIL import Image, ImageOps

accimage = None

class Compose(object):

    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        """
        "Squashing/Composing the transformer chain"
        """

        for transform in self.transforms:
            img = transform(img)

        return img

class ToTensor(object):

    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):

        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            # backward compatibility
            return img.float()

        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))

        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        img = img.view(pic.size[1], pic.size[0], nchannel)

        # put it from HWC to CHW format
        img = img.transpose(0, 1).transpose(0, 2).contiguous()

        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img

class Normalize(object):

        """Normalize an tensor image with mean and standard deviation.
        Given mean: (R, G, B) and std: (R, G, B),
        will normalize each channel of the torch.*Tensor, i.e.
        channel = (channel - mean) / std
        Args:
            mean (sequence): Sequence of means for R, G, B channels respecitvely.
            std (sequence): Sequence of standard deviations for R, G, B channels
                respecitvely.
        """

        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def __call__(self, tensor):

            for transform, mean, std in zip(tensor, self.mean, self.std):
                transform.sub_(mean).div_(std)

            return tensor


class Scale(object):
    """
    Rescale the input PIL.image to the given size
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if self.size is int:
            width, height = img.size

            if (width <= height and width == self.size) or (height <= width and height == self.size):
                return img

        return img.resize(self.size, self.interpolation)

class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):

        if size is numbers.Number:
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        width, height = img.size
        th, tw = self.size
        x1 = int(round((width - tw) / 2.))
        y1 = int(round((height - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))
