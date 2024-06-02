import torch
import torch.nn as nn
from torch.nn import AvgPool1d
from .DiscriminatorS import DiscriminatorS


class MultiScaleDiscriminator(torch.nn.Module):
    """
    Multi-scale discriminator module for HiFi-GAN. This module contains multiple
    spectral discriminators operating at different scales.

    Attributes:
        discriminators (torch.nn.ModuleList): List of spectral discriminators.
        meanpools (torch.nn.ModuleList): List of average pooling layers.
    """

    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, y, y_hat):
        """
        Forward pass of the multi-scale discriminator.

        Args:
            y (torch.Tensor): Real audio tensor.
            y_hat (torch.Tensor): Generated audio tensor.

        Returns:
            tuple: A tuple containing the outputs and feature maps from the real and generated audio.
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
