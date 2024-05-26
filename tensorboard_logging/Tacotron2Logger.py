import matplotlib
import numpy as np
import matplotlib.pylab as plt
import numpy as np
from utils.util import to_arr
from hparams.Tacotron2HParams import Tacotron2HParams as hps
from torch.utils.tensorboard import SummaryWriter
from utils.audio import inv_melspectrogram


matplotlib.use("Agg")


class Tacotron2Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir, flush_secs=5)

    def _save_figure_to_numpy(self, fig):
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data.transpose(2, 0, 1)

    def _plot_alignment_to_numpy(self, alignment, info=None):
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
        fig.colorbar(im, ax=ax)
        xlabel = "Decoder timestep"
        if info is not None:
            xlabel += "\n\n" + info
        plt.xlabel(xlabel)
        plt.ylabel("Encoder timestep")
        plt.tight_layout()

        fig.canvas.draw()
        data = self._save_figure_to_numpy(fig)
        plt.close()
        return data

    def _plot_spectrogram_to_numpy(self, spectrogram):
        fig, ax = plt.subplots(figsize=(12, 3))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=ax)
        plt.xlabel("Frames")
        plt.ylabel("Channels")
        plt.tight_layout()

        fig.canvas.draw()
        data = self._save_figure_to_numpy(fig)
        plt.close()
        return data

    def log_training(self, items, grad_norm, learning_rate, iteration):
        self.add_scalar("loss.mel", items[0], iteration)
        self.add_scalar("loss.gate", items[1], iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)

    def sample_train(self, outputs, iteration):
        mel_outputs = to_arr(outputs[0][0])
        mel_outputs_postnet = to_arr(outputs[1][0])
        alignments = to_arr(outputs[3][0]).T

        # plot alignment, mel and postnet output
        self.add_image(
            "train.align", self._plot_alignment_to_numpy(alignments), iteration
        )
        self.add_image(
            "train.mel", self._plot_spectrogram_to_numpy(mel_outputs), iteration
        )
        self.add_image(
            "train.mel_post",
            self._plot_spectrogram_to_numpy(mel_outputs_postnet),
            iteration,
        )

    def sample_infer(self, outputs, iteration):
        mel_outputs = to_arr(outputs[0][0])
        mel_outputs_postnet = to_arr(outputs[1][0])
        alignments = to_arr(outputs[2][0]).T

        # plot alignment, mel and postnet output
        self.add_image(
            "infer.align", self._plot_alignment_to_numpy(alignments), iteration
        )
        self.add_image(
            "infer.mel", self._plot_spectrogram_to_numpy(mel_outputs), iteration
        )
        self.add_image(
            "infer.mel_post",
            self._plot_spectrogram_to_numpy(mel_outputs_postnet),
            iteration,
        )

        # save audio
        wav = inv_melspectrogram(mel_outputs)
        wav_postnet = inv_melspectrogram(mel_outputs_postnet)
        self.add_audio("infer.wav", wav, iteration, hps.sample_rate)
        self.add_audio("infer.wav_post", wav_postnet, iteration, hps.sample_rate)
