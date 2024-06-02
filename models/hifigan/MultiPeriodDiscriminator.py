import torch
import torch.nn as nn
from .DiscriminatorP import DiscriminatorP


class MultiPeriodDiscriminator(torch.nn.Module):
    """
    Multi-period discriminator module for HiFi-GAN. This module contains multiple
    periodic discriminators, each operating on different periods.

    Attributes:
        discriminators (torch.nn.ModuleList): List of periodic discriminators.
    """

    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, y, y_hat):
        """
        Forward pass of the multi-period discriminator.

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
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
