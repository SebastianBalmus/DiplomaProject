import os
import torch
import logging
from text import text_to_sequence
from models.tacotron2 import Tacotron2
from hparams.Tacotron2HParams import Tacotron2HParams as hps
from utils.util import to_arr
from utils.audio import inv_melspectrogram


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


logger = logging.getLogger(__file__)


class Tacotron2InferenceHandler:
    def __init__(self, args):
        self.args = args.ckpt_pth

        if torch.cuda.is_available():
            torch.cuda.manual_seed(hps.seed)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def _load_model(self):
        ckpt_pth = self.args.ckpt_pth
        assert os.path.isfile(ckpt_pth)
        logger.info(f"Loading checkpoint: {ckpt_pth}")
        ckpt_dict = torch.load(ckpt_pth)
        self.tacotron2 = Tacotron2()
        self.tacotron2.load_state_dict(ckpt_dict["Tacotron2"])
        self.tacotron2 = self.tacotron2.to(self.device)
        self.tacotron2.eval()

    def infer(self, text):
        sequence = text_to_sequence(text, hps.text_cleaners)
        sequence = torch.IntTensor(sequence)[None, :].long().to(self.device)
        mel_outputs, mel_outputs_postnet, _, alignments = self.tacotron2.inference(sequence)
        return (mel_outputs, mel_outputs_postnet, alignments)

    @staticmethod
    def static_infer(text, model, device):
        sequence = text_to_sequence(text, hps.text_cleaners)
        sequence = torch.IntTensor(sequence)[None, :].long().to(device)
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        return (mel_outputs, mel_outputs_postnet, alignments)

    def audio(self, output):
        _, mel_outputs_postnet, _ = output
        wav_postnet = inv_melspectrogram(to_arr(mel_outputs_postnet[0]))
        return wav_postnet
