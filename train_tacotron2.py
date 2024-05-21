import os
import time
import torch
import argparse
import numpy as np
import logging
import torch.multiprocessing as mp
from utils.util import mode
from inference_handlers.Tacotron2InferenceHandler import infer
from hparams.Tacotron2HParams import Tacotron2HParams as hps
from dataset.Tacotron2Dataset import Tacotron2Dataset
from utils.logger import Tacotron2Logger
from models.tacotron2.Tacotron2 import Tacotron2
from models.tacotron2.Loss import Tacotron2Loss
from torch.nn.parallel import DistributedDataParallel

np.random.seed(hps.seed)
torch.manual_seed(hps.seed)
torch.cuda.manual_seed(hps.seed)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

logger = logging.getLogger(__file__)


class Tacotron2Trainer:
    def __init__(self, input_args):
        self.wavs_dir = input_args.wavs_dir
        self.metadata_path = input_args.metadata_path
        self.ckpt_dir = input_args.ckpt_dir
        self.ckpt_path = input_args.ckpt_path
        self.logdir = input_args.logdir
        self.rank = self.local_rank = 0

        if "WORLD_SIZE" in os.environ:
            os.environ["OMP_NUM_THREADS"] = str(hps.n_workers)
            self.rank = int(os.environ["RANK"])
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.num_gpus = int(os.environ["WORLD_SIZE"])
            torch.distributed.init_process_group(
                backend="nccl", rank=self.local_rank, world_size=self.num_gpus
            )
        
        self.device = torch.device("cuda:{:d}".format(self.local_rank))

        self.Tacotron2 = Tacotron2()
        mode(self.Tacotron2, True)

        if self.num_gpus > 1:
            self.Tacotron2 = DistributedDataParallel(
                self.Tacotron2, device_ids=[self.local_rank]
            )
        
        self.optimizer = torch.optim.Adam(
            self.Tacotron2.parameters(),
            lr=hps.lr,
            betas=hps.betas,
            eps=hps.eps,
            weight_decay=hps.weight_decay,
        )

        self.criterion = Tacotron2Loss()

        if self.ckpt_path != "":
            self._load_checkpoint(self.ckpt_dir, self.device)
        else:
            self.epoch = 1

        self._build_env()
        self._create_scheduler()

    def _build_env(self):
        if self.rank == 0:
            logger.info(self.Tacotron2)
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.chmod(self.ckpt_dir, 0o775)
            logger.info(f"Checkpoints directory: {self.ckpt_dir}")
        
            if self.logdir != "":
                if not os.path.isdir(self.logdir):
                    os.makedirs(self.logdir)
                    os.chmod(self.logdir, 0o775)
                self.logger = Tacotron2Logger(self.logdir)
                        
    def _create_scheduler(self, ckpt_path):
        lr_lambda = lambda step: self.hparams.sch_step**0.5 * min(
            (step + 1) * self.hparams.sch_step**-1.5, (step + 1) ** -0.5
        )

        if ckpt_path != "":
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda, last_epoch=self.epoch
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )

    def _load_checkpoint(self, ckpt_path, device):
        assert os.path.isfile(ckpt_path)

        logger.info(f"Loading checkpoint {ckpt_path}")

        ckpt_dict = torch.load(ckpt_path, map_location=device)

        self.Tacotron2.load_state_dict(ckpt_dict["Tacotron2"])
        self.optimizer.load_state_dict(ckpt_dict["optimizer"])
        self.epoch = ckpt_dict["epoch"] + 1

    def _save_checkpoint(self, ckpt_path, num_gpus):
        torch.save(
            dict(
                Tacotron2=(
                    self.Tacotron2.module if num_gpus > 1 else self.Tacotron2
                ).state_dict(),
                optimizer=self.optimizer.state_dict(),
                epoch=self.epoch,
            ),
            ckpt_path,
        )

    def train(self):
        self.train_loader = Tacotron2Dataset.dataloader_factory(
            metadata_path=self.input_args.metadata_path,
            wavs_dir=self.input_args.wavs_dir,
            num_gpus=self.num_gpus,
        )

        self.Tacotron2.train()

        for epoch in range(self.epoch, hps.max_iter + self.epoch):
            if self.num_gpus > 1:
                self.train_loader.sampler.set_epoch(epoch)

            for batch in self.train_loader:
                start = time.perf_counter()

                x, y = (self.Tacotron2.module if self.num_gpus > 1 else self.Tacotron2).parse_batch(batch)

                y_pred = self.Tacotron2(x)

                loss, items = self.criterion(y_pred, y)

                self.optimizer.zero_grad()

                self.loss.backward()


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
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    trainer = Tacotron2Trainer(input_args=args)
    trainer.train()