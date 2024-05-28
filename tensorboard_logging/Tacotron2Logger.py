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
    """
    Tacotron2Logger extends SummaryWriter for logging Tacotron2 training progress to TensorBoard.

    Args:
        logdir (str): Directory for storing TensorBoard logs.
    """

    def __init__(self, logdir):
        super(Tacotron2Logger, self).__init__(logdir, flush_secs=5)

    def _save_figure_to_numpy(self, fig):
        """
        Convert matplotlib figure to numpy array.

        Args:
            fig (matplotlib.figure.Figure): Matplotlib figure to be converted.

        Returns:
            numpy.ndarray: Numpy array representation of the figure.
        """
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data.transpose(2, 0, 1)

    def _plot_alignment_to_numpy(self, alignment, info=None):
        """
        Plot alignment matrix and convert it to numpy array.

        Args:
            alignment (numpy.ndarray): Alignment matrix.
            info (str, optional): Additional information to display in the plot. Defaults to None.

        Returns:
            numpy.ndarray: Numpy array representation of the alignment plot.
        """
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
        """
        Plot spectrogram and convert it to numpy array.

        Args:
            spectrogram (numpy.ndarray): Spectrogram.

        Returns:
            numpy.ndarray: Numpy array representation of the spectrogram plot.
        """
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
        """
        Log training metrics to TensorBoard.

        Args:
            items (tuple): Tuple containing mel loss and gate loss.
            grad_norm (float): Gradient norm.
            learning_rate (float): Learning rate.
            iteration (int): Training iteration.
        """
        self.add_scalar("loss.mel", items[0], iteration)
        self.add_scalar("loss.gate", items[1], iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)

    def sample_train(self, outputs, iteration):
        """
        Log training samples to TensorBoard.

        Args:
            outputs (tuple): Tuple containing mel spectrogram, postnet mel spectrogram, and alignments.
            iteration (int): Training iteration.
        """
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
        """
        Log inference samples to TensorBoard.

        Args:
            outputs (tuple): Tuple containing mel spectrogram, postnet mel spectrogram, and alignments.
            iteration (int): Training iteration.
        """
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
