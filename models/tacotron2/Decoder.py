import torch
from torch.autograd import Variable
from torch.nn import functional as F
from .LinearNorm import LinearNorm
from .Attention import Attention
from .Prenet import Prenet
from hparams.Tacotron2HParams import Tacotron2HParams as hps
from utils.util import get_mask_from_lengths


class Decoder(torch.nn.Module):
    """
    Tacotron2 Decoder module.

    - Prenet for processing decoder inputs.
    - Attention RNN for computing attention context.
    - Attention mechanism for focusing on encoder outputs.
    - Decoder RNN for generating mel-spectrogram frames.
    - Linear projection layer for generating mel outputs.
    - Gate layer for predicting end-of-sequence.

    Args:
        None

    Attributes:
        prenet (Prenet): Prenet for processing decoder inputs.
        attention_rnn (torch.nn.LSTMCell): LSTM cell for attention RNN.
        attention_layer (Attention): Attention mechanism.
        decoder_rnn (torch.nn.LSTMCell): LSTM cell for decoder RNN.
        linear_projection (LinearNorm): Linear layer for generating mel outputs.
        gate_layer (LinearNorm): Linear layer for predicting end-of-sequence gate.
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.prenet = Prenet(
            hps.num_mels * hps.n_frames_per_step, [hps.prenet_dim, hps.prenet_dim]
        )

        self.attention_rnn = torch.nn.LSTMCell(
            hps.prenet_dim + hps.encoder_embedding_dim, hps.attention_rnn_dim
        )

        self.attention_layer = Attention(
            hps.attention_rnn_dim,
            hps.encoder_embedding_dim,
            hps.attention_dim,
            hps.attention_location_n_filters,
            hps.attention_location_kernel_size,
        )

        self.decoder_rnn = torch.nn.LSTMCell(
            hps.attention_rnn_dim + hps.encoder_embedding_dim, hps.decoder_rnn_dim, 1
        )

        self.linear_projection = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim,
            hps.num_mels * hps.n_frames_per_step,
        )

        self.gate_layer = LinearNorm(
            hps.decoder_rnn_dim + hps.encoder_embedding_dim,
            1,
            bias=True,
            w_init_gain="sigmoid",
        )

    def get_go_frame(self, memory):
        """
        Gets all zeros frames to use as first decoder input.

        Args:
            memory (torch.Tensor): Encoder outputs.

        Returns:
            torch.Tensor: All zeros frames of shape (batch_size, num_mels * n_frames_per_step).
        """
        B = memory.size(0)
        decoder_input = Variable(
            memory.data.new(B, hps.num_mels * hps.n_frames_per_step).zero_()
        )
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """
        Initializes the decoder states.

        Args:
            memory (torch.Tensor): Encoder outputs.
            mask (torch.Tensor): Mask for padded data if training, None for inference.
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(
            memory.data.new(B, hps.attention_rnn_dim).zero_()
        )
        self.attention_cell = Variable(
            memory.data.new(B, hps.attention_rnn_dim).zero_()
        )

        self.decoder_hidden = Variable(memory.data.new(B, hps.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, hps.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(
            memory.data.new(B, hps.encoder_embedding_dim).zero_()
        )

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """
        Prepares decoder inputs for training.

        Args:
            decoder_inputs (torch.Tensor): Decoder inputs for teacher-forced training, i.e., mel-spectrograms.

        Returns:
            torch.Tensor: Processed decoder inputs of shape (T_out, batch_size, num_mels * n_frames_per_step).
        """
        # (B, num_mels, T_out) -> (B, T_out, num_mels)
        decoder_inputs = decoder_inputs.transpose(1, 2).contiguous()
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / hps.n_frames_per_step),
            -1,
        )
        # (B, T_out, num_mels) -> (T_out, B, num_mels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """
        Prepares decoder outputs for output.

        Args:
            mel_outputs (list of torch.Tensor): List of mel outputs.
            gate_outputs (list of torch.Tensor): List of gate output energies.
            alignments (list of torch.Tensor): List of attention weights.

        Returns:
            tuple: Tuple containing processed mel outputs, gate outputs, and alignments.
        """
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, num_mels) -> (B, T_out, num_mels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, hps.num_mels)
        # (B, T_out, num_mels) -> (B, num_mels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """
        Decoder step using stored states, attention, and memory.

        Args:
            decoder_input (torch.Tensor): Previous mel output.

        Returns:
            tuple: Tuple containing mel output, gate output, and attention weights.
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        self.attention_hidden = F.dropout(
            self.attention_hidden, hps.p_attention_dropout, self.training
        )

        attention_weights_cat = torch.cat(
            (
                self.attention_weights.unsqueeze(1),
                self.attention_weights_cum.unsqueeze(1),
            ),
            dim=1,
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask,
        )

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, hps.p_decoder_dropout, self.training
        )

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1
        )
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """
        Decoder forward pass for training.

        Args:
            memory (torch.Tensor): Encoder outputs.
            decoder_inputs (torch.Tensor): Decoder inputs for teacher forcing, i.e., mel-spectrograms.
            memory_lengths (torch.Tensor): Encoder output lengths for attention masking.

        Returns:
            tuple: Tuple containing mel outputs, gate outputs, and alignments.
        """
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths)
        )

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze()]
            alignments += [attention_weights]
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """
        Decoder inference.

        Args:
            memory (torch.Tensor): Encoder outputs.

        Returns:
            tuple: Tuple containing mel outputs, gate outputs, and alignments.
        """
        decoder_input = self.get_go_frame(memory)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > hps.gate_threshold:
                print("Terminated by gate.")
                break
            elif (
                hps.n_frames_per_step * len(mel_outputs) / alignment.shape[1]
                >= hps.max_decoder_ratio
            ):
                print("Warning: Reached max decoder steps.")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments
        )
        return mel_outputs, gate_outputs, alignments
