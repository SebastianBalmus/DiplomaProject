import matplotlib
import matplotlib.pylab as plt
from torch.utils.tensorboard import SummaryWriter
from hparams.HiFiGanHParams import HiFiGanHParams as hps


matplotlib.use("Agg")


class HiFiGanLogger(SummaryWriter):
    """
    Custom logger for HiFi-GAN training and validation.

    Args:
        logdir (str): Directory where log files will be written.
    """

    def __init__(self, logdir):
        super(HiFiGanLogger, self).__init__(logdir, flush_secs=5)

    def _plot_spectrogram(self, spectrogram):
        """
        Plot a spectrogram.

        Args:
            spectrogram (numpy.ndarray): Spectrogram to plot.

        Returns:
            matplotlib.figure.Figure: The plotted figure.
        """
        fig, ax = plt.subplots(figsize=(10, 2))
        im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
        plt.colorbar(im, ax=ax)

        fig.canvas.draw()
        plt.close()

        return fig

    def log_training(self, loss_gen_all, mel_error, steps):
        """
        Log training metrics.

        Args:
            loss_gen_all (float): Total generator loss.
            mel_error (float): Mel spectrogram error.
            steps (int): Number of training steps.
        """
        self.add_scalar("training/gen_loss_total", loss_gen_all, steps)
        self.add_scalar("training/mel_spec_error", mel_error, steps)

    def sample_validation(self, x, y, j, steps):
        """
        Log validation samples.

        Args:
            x (torch.Tensor): Ground truth spectrogram.
            y (torch.Tensor): Ground truth audio.
            j (int): Sample index.
            steps (int): Number of training steps.
        """
        self.add_figure("gt/y_spec_{}".format(j), self._plot_spectrogram(x[0]), steps)
        self.add_audio("gt/y_{}".format(j), y[0], steps, hps.sampling_rate)

    def sample_infer(self, y_g_hat, y_hat_spec, j, steps):
        """
        Log generated samples.

        Args:
            y_g_hat (torch.Tensor): Generated audio.
            y_hat_spec (torch.Tensor): Generated spectrogram.
            j (int): Sample index.
            steps (int): Number of training steps.
        """
        self.add_figure(
            "generated/y_hat_spec_{}".format(j),
            self._plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()),
            steps,
        )
        self.add_audio(
            "generated/y_hat_{}".format(j), y_g_hat[0], steps, hps.sampling_rate
        )

    def validation_error(self, val_err, steps):
        """
        Log validation error.

        Args:
            val_err (float): Validation error.
            steps (int): Number of training steps.
        """
        self.add_scalar("validation/mel_spec_error", val_err, steps)
