import os
import torch
import logging
from text import text_to_sequence
from models.tacotron2.Tacotron2 import Tacotron2
from hparams.Tacotron2HParams import Tacotron2HParams as hps
from utils.util import to_arr
from utils.audio import inv_melspectrogram


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


logger = logging.getLogger(__file__)


class Tacotron2InferenceHandler:
    """
    Handler class for Tacotron2 inference.

    Args:
        ckpt_pth (str): Checkpoint path.

    Attributes:
        ckpt_pth (str): Checkpoint path.
        device (torch.device): Device for model inference.
        tacotron2 (Tacotron2): Tacotron2 model instance.
    """

    def __init__(self, ckpt_pth, use_cuda):
        if torch.cuda.is_available() and use_cuda is True:
            torch.cuda.manual_seed(hps.seed)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self._load_model(ckpt_pth)

    def _load_model(self, ckpt_pth):
        """
        Load Tacotron2 model from checkpoint.
        """
        assert os.path.isfile(ckpt_pth)
        logger.info(f"Loading checkpoint: {ckpt_pth}")
        ckpt_dict = torch.load(ckpt_pth, map_location=self.device)
        self.tacotron2 = Tacotron2()
        self.tacotron2.load_state_dict(ckpt_dict["Tacotron2"])
        self.tacotron2 = self.tacotron2.to(self.device)
        self.tacotron2.eval()

    def infer(self, text):
        """
        Perform inference with Tacotron2 model.

        Args:
            text (str): Input text for inference.

        Returns:
            tuple: Tuple containing mel spectrogram outputs, postnet mel spectrogram outputs, and alignments.
        """
        sequence = text_to_sequence(text, hps.text_cleaners)
        sequence = torch.IntTensor(sequence)[None, :].long().to(self.device)
        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron2.inference(sequence)
        return (mel_outputs, mel_outputs_postnet, alignments)

    def infer_e2e(self, text):
        """
        Function used for end to end inference

        Args:
            text (str): Input text for inference.
            model (Tacotron2): Tacotron2 model instance.
            device (torch.device): Device for inference.

        Returns:
            tuple: Tuple containing mel spectrogram outputs, postnet mel spectrogram outputs, and alignments.
        """
        sequence = text_to_sequence(text, hps.text_cleaners)
        sequence = torch.IntTensor(sequence)[None, :].long().to(self.device)
        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron2.inference(sequence)
        return to_arr(mel_outputs_postnet)

    @staticmethod
    def static_infer(text, model, device):
        """
        Perform static inference with Tacotron2 model.

        Args:
            text (str): Input text for inference.
            model (Tacotron2): Tacotron2 model instance.
            device (torch.device): Device for inference.

        Returns:
            tuple: Tuple containing mel spectrogram outputs, postnet mel spectrogram outputs, and alignments.
        """
        sequence = text_to_sequence(text, hps.text_cleaners)
        sequence = torch.IntTensor(sequence)[None, :].long().to(device)
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        return (mel_outputs, mel_outputs_postnet, alignments)

    def audio(self, output):
        """
        Convert mel spectrogram output to audio waveform using the Griffin-Lim algorithm.

        Args:
            output (tuple): Tuple containing mel spectrogram outputs, postnet mel spectrogram outputs, and alignments.

        Returns:
            numpy.ndarray: Audio waveform.
        """
        _, mel_outputs_postnet, _ = output
        wav_postnet = inv_melspectrogram(to_arr(mel_outputs_postnet[0]))
        return wav_postnet


