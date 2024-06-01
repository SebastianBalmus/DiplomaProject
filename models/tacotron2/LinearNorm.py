import torch


class LinearNorm(torch.nn.Module):
    """
    Linear layer with Xavier uniform initialization.

    This module applies a linear transformation to the input data, followed by
    Xavier uniform initialization for the weights.

    Args:
        in_dim (int): The number of input features.
        out_dim (int): The number of output features.
        bias (bool, optional): If set to False, the layer will not learn an additive bias. Default: True.
        w_init_gain (str, optional): The gain to use for Xavier initialization. Default: "linear".

    Attributes:
        linear_layer (torch.nn.Linear): The linear layer that performs the transformation.
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):

        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain)
        )

    def forward(self, x):
        """
        Forward pass through the linear layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """
        return self.linear_layer(x)
