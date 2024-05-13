import torch
import torch.nn.functional as F
from .ConvNorm import ConvNorm
from hparams.Tacotron2HParams import Tacotron2HParams as hps


class Postnet(torch.nn.Module):
    """Postnet
    - Five 1-d convolution with 512 chatorch.nnels and kernel size 5
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
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        return x
