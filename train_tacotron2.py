import os
import time
import torch
import argparse
import numpy as np
import logging
import torch.multiprocessing as mp
from inference_handlers.Tacotron2InferenceHandler import Tacotron2InferenceHandler
from hparams.Tacotron2HParams import Tacotron2HParams as hps
from dataset.Tacotron2Dataset import Tacotron2Dataset
from tensorboard_logging.Tacotron2Logger import Tacotron2Logger
from models.tacotron2.Tacotron2 import Tacotron2
from models.tacotron2.Loss import Tacotron2Loss
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False


class Tacotron2Trainer:
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

        # Generate a random seed for the current GPU
        torch.cuda.manual_seed(self.hparams.seed)

        self.device = torch.device("cuda", self.rank)

        self.Tacotron2 = Tacotron2().to(self.device)

        self._log_to_console(self.Tacotron2)
        os.makedirs(self.input_args.ckpt_dir, exist_ok=True)
        self._log_to_console(f"Checkpoints directory: {self.input_args.ckpt_dir}")

        if self.hparams.num_gpus > 1:
            self.Tacotron2 = DistributedDataParallel(
                self.Tacotron2, device_ids=[self.rank]
            )

        self.optimizer = torch.optim.Adam(
            self.Tacotron2.parameters(),
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
            weight_decay=self.hparams.weight_decay,
        )

        self.loss = Tacotron2Loss()

        if self.input_args.ckpt_path != "":
            self.is_checkpoint = True
            self._load_checkpoint(self.input_args.ckpt_path, self.device)
        else:
            self.is_checkpoint = False
            self.iteration = 1

        self._create_scheduler(self.input_args.ckpt_path)

    def _log_to_console(self, logtext):
        if self.rank == 0:
            self.console_logger.info(logtext)

    def _load_checkpoint(self, ckpt_path, device):
        assert os.path.isfile(ckpt_path)

        self._log_to_console(f"Loading checkpoint {ckpt_path}")

        ckpt_dict = torch.load(ckpt_path, map_location=device)

        self.Tacotron2.load_state_dict(ckpt_dict["Tacotron2"])
        self.optimizer.load_state_dict(ckpt_dict["optimizer"])
        self.iteration = ckpt_dict["iteration"] + 1

    def _save_checkpoint(self, ckpt_path, num_gpus):
        torch.save(
            dict(
                Tacotron2=(
                    self.Tacotron2.module if num_gpus > 1 else self.Tacotron2
                ).state_dict(),
                optimizer=self.optimizer.state_dict(),
                iteration=self.iteration,
            ),
            ckpt_path,
        )

    def _create_scheduler(self, ckpt_path):
        lr_lambda = lambda step: self.hparams.sch_step**0.5 * min(
            (step + 1) * self.hparams.sch_step**-1.5, (step + 1) ** -0.5
        )

        if ckpt_path != "":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda, last_epoch=self.iteration
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )

    def map_array_to_gpu(self, array):
        return map(
            lambda item: item.to(self.device) if torch.is_tensor(item) else item, array
        )

    def train(self):
        self.train_loader = Tacotron2Dataset.dataloader_factory(
            metadata_path=self.input_args.metadata_path,
            wavs_dir=self.input_args.wavs_dir,
            num_gpus=self.hparams.num_gpus,
        )

        if self.rank == 0:
            if self.input_args.logdir != "":
                if not os.path.isdir(self.input_args.logdir):
                    os.makedirs(self.input_args.logdir)
                    os.chmod(self.input_args.logdir, 0o775)
                self.logger = Tacotron2Logger(self.input_args.logdir)

            if self.input_args.ckpt_dir != "" and not os.path.isdir(
                self.input_args.ckpt_dir
            ):
                os.makedirs(self.input_args.ckpt_dir)
                os.chmod(self.input_args.ckpt_dir, 0o775)

        self.Tacotron2.train()

        epoch = 0
        while self.iteration <= self.hparams.max_iter:
            if self.hparams.num_gpus > 1:
                self.train_loader.sampler.set_epoch(epoch)

            for batch in self.train_loader:
                start = time.perf_counter()
                x, y = (
                    self.Tacotron2.module.parse_batch(batch, self.device)
                    if self.hparams.num_gpus > 1
                    else self.Tacotron2.parse_batch(batch, self.device)
                )

                x = self.map_array_to_gpu(x)
                y = self.map_array_to_gpu(y)

                y_pred = self.Tacotron2(x)

                loss, items = self.loss(y_pred, y)

                # zero grad
                self.optimizer.zero_grad()

                # backward, grad_norm, and update
                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.Tacotron2.parameters(), self.hparams.grad_clip_thresh
                )
                self.optimizer.step()
                self.scheduler.step()

                duration = time.perf_counter() - start

                if self.rank == 0:
                    self._log_to_console(
                        f"Iteration: {self.iteration} Mel Loss: {items[0]:.2e} Gate Loss: {items[1]:.2e} Grad Norm: {grad_norm:.2e} {duration:.1f}s/it"
                    )

                    if self.input_args.logdir and (
                        self.iteration % self.hparams.iters_per_log == 0
                    ):
                        learning_rate = self.optimizer.param_groups[0]["lr"]
                        self.logger.log_training(
                            items, grad_norm, learning_rate, self.iteration
                        )

                    if self.iteration % self.hparams.iters_per_sample == 0:
                        self.Tacotron2.eval()
                        output = Tacotron2InferenceHandler.static_infer(
                            self.hparams.eg_text,
                            (
                                self.Tacotron2.module
                                if self.hparams.num_gpus > 1
                                else self.Tacotron2
                            ),
                        )
                        self.Tacotron2.train()
                        self.logger.sample_train(y_pred, self.iteration)
                        self.logger.sample_infer(output, self.iteration)

                    if self.iteration % self.hparams.iters_per_ckpt == 0:
                        ckpt_path = os.path.join(
                            self.input_args.ckpt_dir, f"ckpt_{self.iteration}"
                        )
                        self._save_checkpoint(ckpt_path, self.hparams.num_gpus)

                self.iteration += 1

            self.epoch += 1

        if self.rank == 0 and self.input_args.logdir:
            self.logger.close()

        if self.hparams.num_gpus > 1:
            destroy_process_group()


def multiprocessing_wrapper(rank, input_args, hparams):
    trainer = Tacotron2Trainer(rank=rank, input_args=input_args, hparams=hparams)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-w", "--wavs_dir", type=str, help="Directory where the .wav files are saved"
    )
    parser.add_argument(
        "-m", "--metadata_path", type=str, help="Directory where the metadata is saved"
    )
    parser.add_argument(
        "-cd", "--ckpt_dir", type=str, help="In what directory to save checkpoints"
    )
    parser.add_argument(
        "-cp",
        "--ckpt_path",
        type=str,
        default="",
        help="Path to checkpoint that will be loaded",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="",
        help="Directory where tensorboard logs are saved",
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
        trainer = Tacotron2Trainer(rank=0, input_args=args, hparams=hparams)
        trainer.train()
