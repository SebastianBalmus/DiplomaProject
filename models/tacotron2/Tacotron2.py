import torch
from math import sqrt
from Encoder import Encoder
from Decoder import Decoder
from Postnet import Postnet
from utils.util import mode, get_mask_from_lengths
from hparams.Tacotron2HParams import Tacotron2HParams as hps


class Tacotron2(torch.nn.Module):
    def __init__(self):
        super(Tacotron2, self).__init__()
        self.embedding = torch.nn.Embedding(hps.n_symbols, hps.symbols_embedding_dim)
        std = sqrt(2.0 / (hps.n_symbols + hps.symbols_embedding_dim))
        val = sqrt(3.0) * std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch
        text_padded = mode(text_padded).long()
        input_lengths = mode(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = mode(mel_padded).float()
        gate_padded = mode(gate_padded).float()
        output_lengths = mode(output_lengths).long()
        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded, output_lengths),
        )

    def parse_output(self, outputs, output_lengths=None):
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
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        )
        return outputs

    def teacher_infer(self, inputs, mels):
        il, _ = torch.sort(
            torch.LongTensor([len(x) for x in inputs]), dim=0, descending=True
        )
        text_lengths = mode(il)

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments]
        )
