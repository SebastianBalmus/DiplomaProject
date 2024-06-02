import os
import logging
import torch
from hparams.HiFiGanHParams import HiFiGanHParams as hps
from models.hifigan.Generator import Generator


logger = logging.getLogger(__file__)


class HiFiGanInferenceHandler:
    """
    Handler class for HiFi-GAN inference. It loads a trained HiFi-GAN model and
    performs inference to generate audio from Mel spectrograms.

    Attributes:
        device (torch.device): Device to perform operations on.
        generator (Generator): HiFi-GAN generator model.
    """

    def __init__(self, ckpt_pth, use_cuda):
        if torch.cuda.is_available() and use_cuda is True:
            torch.cuda.manual_seed(hps.seed)
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._load_model(ckpt_pth)

    def _load_model(self, ckpt_pth):
        """
        Load the HiFi-GAN model from a checkpoint file.

        Args:
            ckpt_pth (str): Path to the checkpoint file.

        Raises:
            AssertionError: If the checkpoint file does not exist.
        """
        assert os.path.isfile(ckpt_pth)
        logger.info(f"Loading checkpoint: {ckpt_pth}")

        ckpt_dict = torch.load(ckpt_pth, map_location=self.device)
        self.generator = Generator().to(self.device)
        self.generator.load_state_dict(ckpt_dict["generator"])
        self.generator.eval()

    def infer(self, mel):
        """
        Perform inference to generate audio from a Mel spectrogram.

        Args:
            mel (numpy.array or torch.Tensor): Input Mel spectrogram.

        Returns:
            tuple: A tuple containing the generated audio and the sampling rate.
        """
        self.generator.remove_weight_norm()
        audio = None
        with torch.no_grad():
            x = torch.FloatTensor(mel).to(self.device)
            y_g_hat = self.generator(x)

            audio = y_g_hat.squeeze()
            audio = audio * hps.max_wav_value
            audio = audio.cpu().numpy().astype("int16")

        return (audio, hps.sampling_rate)
