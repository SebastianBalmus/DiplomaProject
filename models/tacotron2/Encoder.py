import torch
from torch.nn import functional as F
from .ConvNorm import ConvNorm
from hparams.Tacotron2HParams import Tacotron2HParams as hps


class Encoder(torch.nn.Module):
    """
    Encoder module for Tacotron2:
    - Three 1-dimensional convolution banks with ReLU activations and dropout.
    - Bidirectional LSTM for sequence modeling.

    Args:
        None

    Attributes:
        convolutions (torch.nn.ModuleList): A list of convolutional layers with batch normalization.
        lstm (torch.nn.LSTM): A bidirectional LSTM layer for processing the output of the convolutional layers.
    """

    def __init__(self):
        super(Encoder, self).__init__()

        convolutions = []
        for i in range(hps.encoder_n_convolutions):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    hps.symbols_embedding_dim if i == 0 else hps.encoder_embedding_dim,
                    hps.encoder_embedding_dim,
                    kernel_size=hps.encoder_kernel_size,
                    stride=1,
                    padding=int((hps.encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                torch.nn.BatchNorm1d(hps.encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)

        self.lstm = torch.nn.LSTM(
            hps.encoder_embedding_dim,
            int(hps.encoder_embedding_dim / 2),
            1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, input_lengths):
        """
        Forward pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, symbols_embedding_dim, sequence_length).
            input_lengths (torch.Tensor): Lengths of the input sequences for packing.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, encoder_embedding_dim).
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs

    def inference(self, x):
        """
        Inference pass through the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, symbols_embedding_dim, sequence_length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, encoder_embedding_dim).
        """
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs
