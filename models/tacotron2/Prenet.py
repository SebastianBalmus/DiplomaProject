import torch
import torch.nn.functional as F
from .LinearNorm import LinearNorm


class Prenet(torch.nn.Module):
    """
    Prenet module for Tacotron2, consisting of a series of linear layers
    followed by ReLU activations and dropout. This module helps in learning
    non-linear representations of the input features.

    Args:
        in_dim (int): The number of input features.
        sizes (list of int): A list of integers defining the number of output
                             features for each linear layer in the prenet.

    Attributes:
        layers (torch.nn.ModuleList): A list of linear layers with specified
                                      input and output sizes.
    """

    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = torch.nn.ModuleList(
            [
                LinearNorm(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_sizes, sizes)
            ]
        )

    def forward(self, x):
        """
        Forward pass through the prenet layers, applying a ReLU activation
        followed by dropout to each linear layer's output.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor after passing through the prenet,
                          with shape (batch_size, sizes[-1]).
        """
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x
