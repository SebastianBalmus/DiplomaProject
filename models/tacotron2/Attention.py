import torch
from torch.nn import functional as F
from .LinearNorm import LinearNorm
from .Location import Location


class Attention(torch.nn.Module):
    """
    Attention mechanism used in Tacotron 2.

    Args:
        attention_rnn_dim (int): Dimensionality of the attention RNN output.
        embedding_dim (int): Dimensionality of the input embeddings (encoder outputs).
        attention_dim (int): Dimensionality of the attention mechanism.
        attention_location_n_filters (int): Number of filters for the location-based attention mechanism.
        attention_location_kernel_size (int): Kernel size for the location-based attention mechanism.

    Attributes:
        query_layer (LinearNorm): Linear layer for processing decoder outputs.
        memory_layer (LinearNorm): Linear layer for processing encoder outputs.
        v (LinearNorm): Linear layer for computing attention energies.
        location_layer (Location): Location-based attention mechanism.
        score_mask_value (float): Value used for masking padded data in attention scores.

    """

    def __init__(
        self,
        attention_rnn_dim,
        embedding_dim,
        attention_dim,
        attention_location_n_filters,
        attention_location_kernel_size,
    ):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(
            attention_rnn_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.memory_layer = LinearNorm(
            embedding_dim, attention_dim, bias=False, w_init_gain="tanh"
        )
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = Location(
            attention_location_n_filters, attention_location_kernel_size, attention_dim
        )
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory, attention_weights_cat):
        """
        Compute attention energies.

        Args:
            query (torch.Tensor): Decoder output tensor of shape (batch_size, decoder_dim).
            processed_memory (torch.Tensor): Processed encoder outputs tensor of shape (batch_size, max_time, attention_dim).
            attention_weights_cat (torch.Tensor): Concatenated previous and cumulative attention weights tensor of shape (batch_size, 2, max_time).

        Returns:
            torch.Tensor: Attention energies tensor of shape (batch_size, max_time).

        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights + processed_memory)
        )

        energies = energies.squeeze(-1)
        return energies

    def forward(
        self,
        attention_hidden_state,
        memory,
        processed_memory,
        attention_weights_cat,
        mask,
    ):
        """
        Compute attention context vector and attention weights.

        Args:
            attention_hidden_state (torch.Tensor): Last output of the attention RNN of shape (batch_size, attention_rnn_dim).
            memory (torch.Tensor): Encoder outputs tensor of shape (batch_size, max_time, embedding_dim).
            processed_memory (torch.Tensor): Processed encoder outputs tensor of shape (batch_size, max_time, attention_dim).
            attention_weights_cat (torch.Tensor): Concatenated previous and cumulative attention weights tensor of shape (batch_size, 2, max_time).
            mask (torch.Tensor): Binary mask tensor for padded data of shape (batch_size, max_time).

        Returns:
            torch.Tensor: Attention context vector of shape (batch_size, embedding_dim).
            torch.Tensor: Attention weights tensor of shape (batch_size, max_time).

        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat
        )

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)
        return attention_context, attention_weights
