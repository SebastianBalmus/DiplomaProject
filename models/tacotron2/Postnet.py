import torch
import torch.nn.functional as F
from .ConvNorm import ConvNorm
from hparams.Tacotron2HParams import Tacotron2HParams as hps


class Postnet(torch.nn.Module):
    """
    Postnet
    - A sequence of five 1-dimensional convolutional layers with batch
      normalization and dropout, as used in the Tacotron2 architecture.
    - The first and last layers have different input and output dimensions
      compared to the intermediate layers.

    Args:
        None

    Attributes:
        convolutions (torch.nn.ModuleList): A list of sequential layers, each
                                            containing a ConvNorm layer followed
                                            by BatchNorm1d.
    """

    def __init__(self):
        super(Postnet, self).__init__()
        self.convolutions = torch.nn.ModuleList()

        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(
                    hps.num_mels,
                    hps.postnet_embedding_dim,
                    kernel_size=hps.postnet_kernel_size,
                    stride=1,
                    padding=int((hps.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                torch.nn.BatchNorm1d(hps.postnet_embedding_dim),
            )
        )

        for i in range(1, hps.postnet_n_convolutions - 1):
            self.convolutions.append(
                torch.nn.Sequential(
                    ConvNorm(
                        hps.postnet_embedding_dim,
                        hps.postnet_embedding_dim,
                        kernel_size=hps.postnet_kernel_size,
                        stride=1,
                        padding=int((hps.postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    torch.nn.BatchNorm1d(hps.postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(
                    hps.postnet_embedding_dim,
                    hps.num_mels,
                    kernel_size=hps.postnet_kernel_size,
                    stride=1,
                    padding=int((hps.postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                torch.nn.BatchNorm1d(hps.num_mels),
            )
        )

    def forward(self, x):
        """
        Forward pass through the Postnet layers. Applies a series of convolutional
        layers with tanh activation and dropout, except for the last layer which
        uses a linear activation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_mels, time_steps).

        Returns:
            torch.Tensor: Output tensor after passing through the Postnet layers,
                          with shape (batch_size, num_mels, time_steps).
        """
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x
