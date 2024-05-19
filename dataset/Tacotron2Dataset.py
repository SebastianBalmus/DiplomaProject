import os
import csv
import torch
import logging
import numpy as np
from tqdm import tqdm
from text import text_to_sequence
from hparams.Tacotron2HParams import Tacotron2HParams as hps
from torch.utils.data import Dataset, DistributedSampler, DataLoader
from utils.audio import load_wav, melspectrogram


logger = logging.getLogger(__name__)


class Tacotron2Dataset(Dataset):
    def __init__(self, metadata_path, wavs_dir, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step
        self._read_metadata(metadata_path)

        self.data = []

        for id_text_pair in tqdm(
            self.metadata,
            desc='Loading dataset',
            mininters=int(len(self.metadata) / 200),
            leave=True,
            position=0
        ):
            assert len(id_text_pair) >= 2
            id = id_text_pair[0]
            text = id_text_pair[1]
            wav_path = os.path.join(wavs_dir, f"{id}.wav")

            text_mel_pair = self._get_text_mel_pair(text, wav_path)

            self.data.append(text_mel_pair)

    def _read_metadata(self, metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="|")
            self.metadata = list(reader)

    def _get_text_mel_pair(self, text, wav_path):
        text = torch.IntTensor(text_to_sequence(text, hps.text_cleaners))
        wav = load_wav(wav_path)
        mel = torch.Tensor(melspectrogram(wav).astype(np.float32))

        return dict(text=text, mel=mel)

    def __getitem__(self, index):
        text_mel_pair = self.data[index]
        text = text_mel_pair["text"]
        mel = text_mel_pair["mel"]

        return text, mel

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, : text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += (
                self.n_frames_per_step - max_target_len % self.n_frames_per_step
            )
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, : mel.size(1)] = mel
            gate_padded[i, mel.size(1) - 1 :] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths

    @staticmethod
    def dataloader_factory(metadata_path, wavs_dir, num_gpus):
        dataset = Tacotron2Dataset(
            metadata_path=metadata_path,
            wavs_dir=wavs_dir,
            n_frames_per_step=hps.n_frames_per_step
        )
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None

        data_loader = DataLoader(
            dataset,
            num_workers=hps.n_workers,
            shuffle=num_gpus == 1,
            batch_size=hps.batch_size,
            pin_memory=hps.pin_mem,
            drop_last=True,
            collate_fn=dataset.collate_fn,
            sampler=sampler,
        )

        return data_loader

