import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import itertools
import os
import time
import argparse
import json
import logging
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from dataset.HiFiGanDataset import HiFiGanDataset
from models.hifigan.Generator import Generator
from models.hifigan.MultiPeriodDiscriminator import MultiPeriodDiscriminator
from models.hifigan.MultiScaleDiscriminator import MultiScaleDiscriminator
from models.hifigan.Loss import feature_loss, generator_loss, discriminator_loss
from tensorboard_logging.HiFiGanLogger import HiFiGanLogger
from hparams.HiFiGanHParams import HiFiGanHParams as hps


torch.backends.cudnn.benchmark = True


class HiFiGanTrainer:
    def __init__(self, rank, input_args, hparams):
        self.rank = rank
        self.input_args = input_args
        self.hparams = hparams

        if self.hparams.num_gpus > 1:
            init_process_group(
                backend=self.hparams.dist_backend,
                init_method=self.hparams.dist_url,
                world_size=self.hparams.world_size * self.hparams.num_gpus,
                rank=self.rank,
            )

        if self.rank == 0:
            logging.basicConfig(level=logging.INFO)
            self.console_logger = logging.getLogger()

        torch.cuda.manual_seed(self.hparams.seed)
        self.device = torch.device("cuda", self.rank)

        self.generator = Generator(self.hparams).to(self.device)
        self.mpd = MultiPeriodDiscriminator().to(self.device)
        self.msd = MultiScaleDiscriminator().to(self.device)

        self._log_to_console(self.generator)
        self._log_to_console(f"Checkpoints directory: {self.input_args.ckpt_dir}")

        if self.hparams.num_gpus > 1:
            self.generator = DistributedDataParallel(
                self.generator, device_ids=[self.rank]
            ).to(self.device)
            self.mpd = DistributedDataParallel(self.mpd, device_ids=[self.rank]).to(
                self.device
            )
            self.msd = DistributedDataParallel(self.msd, device_ids=[self.rank]).to(
                self.device
            )

        self.optim_g = torch.optim.AdamW(
            self.generator.parameters(),
            self.hparams.learning_rate,
            betas=self.hparams.betas,
        )
        self.optim_d = torch.optim.AdamW(
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
            self.hparams.learning_rate,
            betas=self.hparams.betas,
        )

        if (
            self.input_args.ckpt_path_generator != ""
            and self.input_args.ckpt_path_discriminator != ""
        ):
            self._load_checkpoint(
                self.input_args.ckpt_path_generator,
                self.input_args.ckpt_path_discriminator,
                self.device,
            )
        else:
            self.last_epoch = -1

        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g, gamma=self.hparams.lr_decay, last_epoch=self.last_epoch
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d, gamma=self.hparams.lr_decay, last_epoch=self.last_epoch
        )

    def _log_to_console(self, logtext):
        """
        Logs messages to the console if the process rank is 0. This ensures that only
        one process outputs log messages in a multi-GPU setting.

        Args:
            logtext (str): The message to be logged.
        """
        if self.rank == 0:
            self.console_logger.info(logtext)

    def _load_checkpoint(self, ckpt_path_generator, ckpt_path_discriminator, device):
        assert os.path.isfile(ckpt_path_generator)
        assert os.path.isfile(ckpt_path_discriminator)

        self._log_to_console(f"Loading generator checkpoint: {ckpt_path_generator}")
        self._log_to_console(
            f"Loading discriminator checkpoint: {ckpt_path_discriminator}"
        )

        ckpt_dict_generator = torch.load(ckpt_path_generator, map_location=device)
        ckpt_dict_discriminator = torch.load(
            ckpt_path_discriminator, map_location=device
        )

        self.generator.load_state_dict(ckpt_dict_generator["generator"])
        self.mpd.load_state_dict(ckpt_dict_discriminator["mpd"])
        self.msd.load_state_dict(ckpt_dict_discriminator["msd"])

        self.optim_g.load_state_dict(ckpt_dict_generator["optim_g"])
        self.optim_d.load_state_dict(ckpt_dict_discriminator["optim_d"])

        self.steps = ckpt_dict_discriminator["steps"] + 1
        self.last_epoch = ckpt_dict_discriminator["epoch"]

    def _save_checkpoint(self, ckpt_path_generator, ckpt_path_discriminator, epoch):
        torch.save(
            {
                "generator": (
                    self.generator.module
                    if self.hparams.num_gpus > 1
                    else self.generator
                ).state_dict(),
                "optim_g": self.optim_g.state_dict(),
            },
            ckpt_path_generator,
        )
        torch.save(
            {
                "mpd": (
                    self.mpd.module if self.hparams.num_gpus > 1 else self.mpd
                ).state_dict(),
                "msd": (
                    self.msd.module if self.hparams.num_gpus > 1 else self.msd
                ).state_dict(),
                "optim_d": self.optim_d.state_dict(),
                "steps": self.steps,
                "epoch": epoch,
            },
            ckpt_path_discriminator,
        )

    def train(self):
        # Setup logging and checkpoint directories if this is the master process
        if self.rank == 0:
            if self.input_args.logdir != "":
                if not os.path.isdir(self.input_args.logdir):
                    os.makedirs(self.input_args.logdir)
                    os.chmod(self.input_args.logdir, 0o775)
                self.logger = HiFiGanLogger(self.input_args.logdir)

            if self.input_args.ckpt_dir != "" and not os.path.isdir(
                self.input_args.ckpt_dir
            ):
                os.makedirs(self.input_args.ckpt_dir)
                os.chmod(self.input_args.ckpt_dir, 0o775)

        training_filelist, validation_filelist = HiFiGanDataset.get_dataset_filelist(
            input_training_file=self.input_args.input_training_file,
            input_validation_file=self.input_args.input_validation_file,
            input_wavs_dir=self.input_args.input_wavs_dir,
        )

        self.train_loader = HiFiGanDataset.dataloader_factory(
            filelist=training_filelist,
            device=self.device,
            fine_tuning=self.input_args.fine_tuning,
            input_mels_dir=self.input_args.input_mels_dir,
            num_gpus=self.hparams.num_gpus,
            validation=False,
        )

        if self.rank == 0:
            self.validation_loader = HiFiGanDataset.dataloader_factory(
                filelist=validation_filelist,
                device=self.device,
                fine_tuning=self.input_args.fine_tuning,
                input_mels_dir=self.input_args.input_mels_dir,
                num_gpus=self.hparams.num_gpus,
                validation=True,
            )

        self.generator.train()
        self.mpd.train()
        self.msd.train()

        self.steps = 1

        for epoch in range(max(0, self.last_epoch), self.hparams.training_epochs):
            if self.rank == 0:
                start = time.time()
                self._log_to_console(f"Epoch {epoch + 1}")

            if self.hparams.num_gpus > 1:
                self.train_loader.sampler.set_epoch(epoch)

            for i, batch in enumerate(self.train_loader):
                if self.rank == 0:
                    start_b = time.time()

                x, y, _, y_mel = batch

                x = torch.autograd.Variable(x.to(self.device, non_blocking=True))
                y = torch.autograd.Variable(y.to(self.device, non_blocking=True))
                y_mel = torch.autograd.Variable(
                    y_mel.to(self.device, non_blocking=True)
                )
                y = y.unsqueeze(1)

                y_g_hat = self.generator(x)
                y_g_hat_mel = HiFiGanDataset.mel_spectrogram(
                    y=y_g_hat.squeeze(1),
                    n_fft=self.hparams.n_fft,
                    num_mels=self.hparams.num_mels,
                    sampling_rate=self.hparams.sampling_rate,
                    hop_size=self.hparams.hop_size,
                    win_size=self.hparams.win_size,
                    fmin=self.hparams.fmin,
                    fmax=self.hparams.fmax_for_loss,
                )

                self.optim_d.zero_grad()

                # MPD
                y_df_hat_r, y_df_hat_g, _, _ = self.mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = self.msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                self.optim_d.step()

                # Generator
                self.optim_g.zero_grad()

                # L1 Mel-Spectrogram Loss
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
                loss_gen_all = (
                    loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
                )

                loss_gen_all.backward()
                self.optim_g.step()

                if self.rank == 0:
                    # STDOUT logging
                    if self.steps % self.hparams.stdout_interval == 0:
                        with torch.no_grad():
                            mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                        self._log_to_console(
                            f"Steps : {self.steps}, Gen Loss Total : {loss_gen_all}, Mel-Spec. Error : {mel_error}, s/b : {time.time() - start_b}"
                        )

                    if (
                        self.steps % self.hparams.checkpoint_interval == 0
                        and self.steps != 0
                    ):
                        generator_ckpt_path = "{}/g_{:08d}".format(
                            self.input_args.ckpt_dir, self.steps
                        )
                        discriminator_ckpt_path = "{}/do_{:08d}".format(
                            self.input_args.ckpt_dir, self.steps
                        )
                        self._save_checkpoint(
                            ckpt_path_generator=generator_ckpt_path,
                            ckpt_path_discriminator=discriminator_ckpt_path,
                            epoch=epoch,
                        )

                    # Tensorboard logging
                    if self.steps % self.hparams.summary_interval == 0:
                        self.logger.log_training(
                            loss_gen_all=loss_gen_all,
                            mel_error=mel_error,
                            steps=self.steps,
                        )

                    # Validation
                    if self.steps % self.hparams.validation_interval == 0:
                        self.generator.eval()
                        torch.cuda.empty_cache()
                        val_err_tot = 0

                        with torch.no_grad():
                            for j, val_batch in enumerate(self.validation_loader):
                                x, y, _, y_mel = val_batch
                                y_g_hat = self.generator(x.to(self.device))
                                y_mel = torch.autograd.Variable(
                                    y_mel.to(self.device, non_blocking=True)
                                )
                                y_g_hat_mel = HiFiGanDataset.mel_spectrogram(
                                    y_g_hat.squeeze(1),
                                    self.hparams.n_fft,
                                    self.hparams.num_mels,
                                    self.hparams.sampling_rate,
                                    self.hparams.hop_size,
                                    self.hparams.win_size,
                                    self.hparams.fmin,
                                    self.hparams.fmax_for_loss,
                                )
                                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                                if j <= 4:
                                    if self.steps == 0:
                                        self.logger.sample_validation(
                                            x, y, j, self.steps
                                        )

                                    y_hat_spec = HiFiGanDataset.mel_spectrogram(
                                        y_g_hat.squeeze(1),
                                        self.hparams.n_fft,
                                        self.hparams.num_mels,
                                        self.hparams.sampling_rate,
                                        self.hparams.hop_size,
                                        self.hparams.win_size,
                                        self.hparams.fmin,
                                        self.hparams.fmax_for_loss,
                                    )

                                    self.logger.sample_infer(
                                        y_g_hat, y_hat_spec, j, self.steps
                                    )

                            val_err = val_err = val_err_tot / (j + 1)
                            self.logger.validation_error(val_err, self.steps)

                        self.generator.train()

                self.steps += 1

            self.scheduler_g.step()
            self.scheduler_d.step()

            if self.rank == 0:
                self._log_to_console(
                    f"Time taken for epoch {epoch + 1} is {int(time.time() - start)}"
                )

        if self.rank == 0 and self.input_args.logdir:
            self.logger.close()

        if self.hparams.num_gpus > 1:
            destroy_process_group()


def multiprocessing_wrapper(rank, input_args, hparams):
    trainer = HiFiGanTrainer(rank=rank, input_args=input_args, hparams=hparams)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w",
        "--input_wavs_dir",
        type=str,
        help="Directory where the .wav files are saved",
    )
    parser.add_argument(
        "-m",
        "--input_mels_dir",
        type=str,
        default="",
        help="Directory where the mel spectrograms are saved",
    )
    parser.add_argument(
        "-t", "--input_training_file", type=str, help="Training metadata.csv file"
    )
    parser.add_argument(
        "-v", "--input_validation_file", type=str, help="Validation metadata.csv file"
    )
    parser.add_argument(
        "-cd", "--ckpt_dir", type=str, help="In what directory to save checkpoints"
    )
    parser.add_argument(
        "--ckpt_path_generator", type=str, default="", help="Generator checkpoint path"
    )
    parser.add_argument(
        "--ckpt_path_discriminator",
        type=str,
        default="",
        help="Discriminator checkpoint path",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="",
        help="Directory where tensorboard logs are saved",
    )
    parser.add_argument(
        "--fine_tuning",
        type=bool,
        default=False,
        help="Fine tune an existing or start from scratch",
    )

    args = parser.parse_args()
    hparams = hps

    if torch.cuda.is_available():
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        hparams.num_gpus = torch.cuda.device_count()
        hparams.batch_size = hparams.batch_size // hparams.num_gpus

    if hparams.num_gpus > 1:
        mp.spawn(multiprocessing_wrapper, nprocs=hparams.num_gpus, args=(args, hparams))
    else:
        trainer = HiFiGanTrainer(rank=0, input_args=args, hparams=hparams)
        trainer.train()
