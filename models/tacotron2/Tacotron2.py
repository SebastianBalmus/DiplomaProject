import torch
from math import sqrt
from .Encoder import Encoder
from .Decoder import Decoder
from .Postnet import Postnet
from utils.util import get_mask_from_lengths
from hparams.Tacotron2HParams import Tacotron2HParams as hps


class Tacotron2(torch.nn.Module):
    """Tacotron 2 Model

    A PyTorch implementation of the Tacotron 2 model for text-to-speech synthesis.

    Attributes:
        embedding (torch.nn.Embedding): Embedding layer for text input.
        encoder (Encoder): Encoder component of the Tacotron2 model.
        decoder (Decoder): Decoder component of the Tacotron2 model.
        postnet (Postnet): Postnet component of the Tacotron2 model.
    """

    def __init__(self):
        super(Tacotron2, self).__init__()
        self.embedding = torch.nn.Embedding(hps.n_symbols, hps.symbols_embedding_dim)
        std = sqrt(2.0 / (hps.n_symbols + hps.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def parse_batch(self, batch, device):
        """
        Parses a batch of data and transfers it to the specified device.

        Args:
            batch (tuple): A tuple containing the following elements:
                - text_padded (torch.Tensor): Padded text sequences.
                - input_lengths (torch.Tensor): Lengths of the input sequences.
                - mel_padded (torch.Tensor): Padded mel spectrograms.
                - gate_padded (torch.Tensor): Padded gate values.
                - output_lengths (torch.Tensor): Lengths of the output sequences.
            device (torch.device): The device to which the tensors should be transferred.

        Returns:
            tuple: A tuple containing two elements:
                - tuple: A tuple containing:
                    - text_padded (torch.Tensor): Padded text sequences transfered on the specified device.
                    - input_lengths (torch.Tensor): Lengths of the input sequences transfered on the specified device.
                    - max_len (int): Maximum length of the input sequences.
                    - mel_padded (torch.Tensor): Padded mel spectrograms transfered on the specified device.
                    - output_lengths (torch.Tensor): Lengths of the output sequences transfered on the specified device.
                - tuple: A tuple containing:
                    - mel_padded (torch.Tensor): Padded mel spectrograms transfered on the specified device.
                    - gate_padded (torch.Tensor): Padded gate values transfered on the specified device.
                    - output_lengths (torch.Tensor): Lengths of the output sequences transfered on the specified device.
        """
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = text_padded.long().to(device)
        input_lengths = input_lengths.long().to(device)
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mel_padded.float().to(device)
        gate_padded = gate_padded.float().to(device)
        output_lengths = output_lengths.long().to(device)
        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded, output_lengths),
        )

    def parse_output(self, outputs, output_lengths=None):
        """
        Parses and processes the model outputs, applying a mask if output lengths are provided.

        Args:
            outputs (list of torch.Tensor): A list containing the following elements:
                - torch.Tensor: Predicted mel spectrograms.
                - torch.Tensor: Predicted postnet mel spectrograms.
                - torch.Tensor: Predicted gate values.
            output_lengths (torch.Tensor, optional): Lengths of the output sequences. Defaults to None.

        Returns:
            list of torch.Tensor: The processed outputs with masks applied if output_lengths is provided.
        """
        if output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, True)  # (B, T)
            mask = mask.expand(hps.num_mels, mask.size(0), mask.size(1))  # (80, B, T)
            mask = mask.permute(1, 0, 2)  # (B, 80, T)

            outputs[0].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            outputs[1].data.masked_fill_(mask, 0.0)  # (B, 80, T)
            slice = torch.arange(0, mask.size(2), hps.n_frames_per_step)
            outputs[2].data.masked_fill_(
                mask[:, 0, slice], 1e3
            )  # gate energies (B, T//n_frames_per_step)
        return outputs

    def forward(self, inputs):
        """
        Forward pass for the model.

        Args:
            inputs (tuple): A tuple containing the following elements:
                - text_inputs (torch.Tensor): Padded text sequences.
                - text_lengths (torch.Tensor): Lengths of the text sequences.
                - mels (torch.Tensor): Padded mel spectrograms.
                - max_len (int): Maximum length of the input sequences.
                - output_lengths (torch.Tensor): Lengths of the output sequences.

        Returns:
            list of torch.Tensor: The processed outputs including:
                - mel_outputs (torch.Tensor): Predicted mel spectrograms.
                - mel_outputs_postnet (torch.Tensor): Refined mel spectrograms after postnet.
                - gate_outputs (torch.Tensor): Predicted gate values.
                - alignments (torch.Tensor): Attention alignments.
        """
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments], output_lengths
        )

    def inference(self, inputs):
        """
        Performs inference with the Tacotron2 model.

        Args:
            inputs (torch.Tensor): Input text sequences.

        Returns:
            list of torch.Tensor: The processed outputs including:
                - mel_outputs (torch.Tensor): Predicted mel spectrograms.
                - mel_outputs_postnet (torch.Tensor): Refined mel spectrograms after postnet.
                - gate_outputs (torch.Tensor): Predicted gate values.
                - alignments (torch.Tensor): Attention alignments.
        """
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        )
        return outputs

    def teacher_inference(self, inputs, mels):
            """
        Performs teacher-forced inference with the Tacotron2 model.

        This method generates mel-spectrograms using both input text sequences 
        and corresponding ground truth mel-spectrograms. It prepares the input 
        sequences, encodes them, and processes the mel-spectrograms through the 
        decoder and postnet with teacher forcing.

        Args:
            inputs (torch.Tensor): Input text sequences.
            mels (torch.Tensor): Ground truth mel-spectrograms.

        Returns:
            list of torch.Tensor: The processed outputs including:
                - mel_outputs (torch.Tensor): Predicted mel spectrograms.
                - mel_outputs_postnet (torch.Tensor): Refined mel spectrograms after postnet.
                - gate_outputs (torch.Tensor): Predicted gate values.
                - alignments (torch.Tensor): Attention alignments.
        """
        text_lengths, _ =  torch.sort(torch.LongTensor([len(x) for x in inputs]).to('cuda'),
                            dim = 0, descending = True)

        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        outputs =  self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])
        return outputs
