import torch
import numpy as np
from hparams.Tacotron2HParams import Tacotron2HParams as hps


def to_arr(var):
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        var (torch.Tensor): Input PyTorch tensor.

    Returns:
        numpy.ndarray: NumPy array representation of the tensor.
    """
    return var.cpu().detach().numpy().astype(np.float32)


def get_mask_from_lengths(lengths, pad=False):
    """
    Generate a mask tensor based on sequence lengths.

    Args:
        lengths (torch.Tensor): Tensor containing sequence lengths.
        pad (bool, optional): Whether to pad sequences to ensure uniform length. Defaults to False.

    Returns:
        torch.Tensor: Mask tensor with shape (batch_size, max_length), where max_length is the maximum sequence length.
    """
    max_len = torch.max(lengths).item()
    if pad and max_len % hps.n_frames_per_step != 0:
        max_len += hps.n_frames_per_step - max_len % hps.n_frames_per_step
        assert max_len % hps.n_frames_per_step == 0
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0)

    mask = ids < lengths.unsqueeze(1)
    return mask

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)

