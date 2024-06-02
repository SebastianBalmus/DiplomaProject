import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, spectral_norm
from hparams.HiFiGanHParams import HiFiGanHParams as hps


class DiscriminatorS(torch.nn.Module):
    """
    Spectral discriminator module for HiFi-GAN. This discriminator operates on the
    spectral representation of the input audio.

    Attributes:
        convs (torch.nn.ModuleList): Convolutional layers of the discriminator.
        conv_post (torch.nn.Conv1d): Final convolutional layer.
    """

    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Forward pass of the spectral discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, timesteps).

        Returns:
            tuple: A tuple containing the output tensor and feature maps.
        """
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, hps.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap
