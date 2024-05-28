import torch


class ConvNorm(torch.nn.Module):
    """
    1D Convolutional Layer with optional batch normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 1.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding added to both sides of the input. If None, padding is calculated based on the kernel size and dilation. Defaults to None.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
        w_init_gain (str, optional): Type of weight initialization gain. Defaults to "linear".

    Attributes:
        conv (torch.nn.Conv1d): Convolutional layer.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):

        super(ConvNorm, self).__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, signal):
        """
        Forward pass through the ConvNorm layer.

        Args:
            signal (torch.Tensor): Input tensor of shape (batch_size, in_channels, signal_length).

        Returns:
            torch.Tensor: Output tensor after convolution of shape (batch_size, out_channels, output_length).
        """
        conv_signal = self.conv(signal)
        return conv_signal
