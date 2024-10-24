import torch
from utils.util import get_mask_from_lengths
from hparams.Tacotron2HParams import Tacotron2HParams as hps


class Tacotron2Loss(torch.nn.Module):
    """
    Loss function for Tacotron 2 model.

    Computes the combined loss for mel spectrogram prediction and stop token prediction.

    Attributes:
        loss (torch.nn.MSELoss): Mean squared error loss function for mel spectrogram prediction.

    """

    def __init__(self):
        super(Tacotron2Loss, self).__init__()
        self.loss = torch.nn.MSELoss(reduction="none")

    def forward(self, model_outputs, targets):
        """
        Compute the loss for Tacotron 2 model.

        Args:
            model_outputs (tuple): Tuple containing model outputs including mel spectrogram predictions, postnet mel spectrogram predictions, gate predictions, and alignment predictions.
            targets (tuple): Tuple containing target values including target mel spectrograms, gate targets, and output lengths.

        Returns:
            torch.Tensor: Combined loss value.
            tuple: Tuple containing individual losses for mel spectrogram prediction and gate prediction.

        """
        mel_out, mel_out_postnet, gate_out, _ = model_outputs
        gate_out = gate_out.view(-1, 1)

        mel_target, gate_target, output_lengths = targets
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        output_lengths.requires_grad = False
        slice = torch.arange(0, gate_target.size(1), hps.n_frames_per_step)
        gate_target = gate_target[:, slice].view(-1, 1)
        mel_mask = ~get_mask_from_lengths(output_lengths.data, True)

        mel_loss = self.loss(mel_out, mel_target) + self.loss(
            mel_out_postnet, mel_target
        )
        mel_loss = mel_loss.sum(1).masked_fill_(mel_mask, 0.0) / mel_loss.size(1)
        mel_loss = mel_loss.sum() / output_lengths.sum()

        gate_loss = torch.nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss, (mel_loss.item(), gate_loss.item())
