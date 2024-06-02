import torch
import numpy as np


def init_weights(m, mean=0.0, std=0.01):
    """
    Initialize the weights of the module.

    Args:
        m (torch.nn.Module): The module to initialize.
        mean (float): Mean of the normal distribution for weight initialization.
        std (float): Standard deviation of the normal distribution for weight initialization.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """
    Calculate padding size for a convolutional layer.

    Args:
        kernel_size (int): Size of the convolutional kernel.
        dilation (int): Dilation rate of the convolutional layer.

    Returns:
        int: Padding size.
    """
    return int((kernel_size * dilation - dilation) / 2)
