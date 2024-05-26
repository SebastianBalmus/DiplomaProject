import torch
from .ConvNorm import ConvNorm
from .LinearNorm import LinearNorm


class Location(torch.nn.Module):
    """
    Location-sensitive attention mechanism for Tacotron2.

    This module processes the cumulative attention weights through a series
    of convolutional and linear layers to produce location-based features
    that aid the attention mechanism in focusing on the correct part of the
    input sequence.

    Args:
        attention_n_filters (int): Number of filters for the convolutional layer.
        attention_kernel_size (int): Kernel size for the convolutional layer.
        attention_dim (int): Dimension of the output linear transformation.

    Attributes:
        location_conv (ConvNorm): Convolutional layer that processes the attention weights.
        location_dense (LinearNorm): Linear layer that processes the output of the convolutional layer.
    """

    def __init__(self, attention_n_filters, attention_kernel_size, attention_dim):
        super(Location, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(
            2,
            attention_n_filters,
            kernel_size=attention_kernel_size,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
        )
        self.location_dense = LinearNorm(
            attention_n_filters, attention_dim, bias=False, w_init_gain="tanh"
        )

    def forward(self, attention_weights_cat):
        """
        Forward pass through the Location-sensitive attention mechanism.

        Args:
            attention_weights_cat (torch.Tensor): Concatenated previous and
                                                  cumulative attention weights of shape
                                                  (batch_size, 2, max_time).

        Returns:
            torch.Tensor: Processed location features of shape
                          (batch_size, max_time, attention_dim).
        """
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention
