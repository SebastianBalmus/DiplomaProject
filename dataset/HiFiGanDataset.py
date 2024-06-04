import math
import os
import random
import torch
from torch.utils.data import DistributedSampler, DataLoader
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from hparams.HiFiGanHParams import HiFiGanHParams as hps


class HiFiGanDataset:
    """
    Dataset class for HiFi-GAN model. It handles loading, processing, and
    augmentation of audio data for training and evaluation.

    Attributes:
        audio_files (list): List of paths to audio files.
        segment_size (int): Size of the audio segments to be used.
        sampling_rate (int): Sampling rate of the audio data.
        split (bool): Whether to split the audio into segments.
        n_fft (int): Number of FFT components.
        num_mels (int): Number of Mel bands.
        hop_size (int): Hop size for STFT.
        win_size (int): Window size for STFT.
        fmin (int): Minimum frequency for Mel filterbank.
        fmax (int): Maximum frequency for Mel filterbank.
        fmax_loss (int, optional): Maximum frequency for Mel filterbank used in loss calculation.
        cached_wav (numpy array): Cached waveform data.
        n_cache_reuse (int): Number of times to reuse the cached waveform.
        device (torch.device): Device to perform operations on.
        fine_tuning (bool): Whether the dataset is used for fine-tuning.
        base_mels_path (str): Base path for Mel spectrograms when fine-tuning.
    """

    def __init__(
        self,
        training_files,
        segment_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        split=True,
        shuffle=True,
        n_cache_reuse=1,
        device=None,
        fmax_loss=None,
        fine_tuning=False,
        base_mels_path=None,
    ):
        self.audio_files = training_files
        random.seed(1234)

        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    @staticmethod
    def _load_wav(full_path):
        """
        Load a WAV file.

        Args:
            full_path (str): Path to the WAV file.

        Returns:
            tuple: A tuple containing the audio data and the sampling rate.
        """
        sampling_rate, data = read(full_path)
        return data, sampling_rate

    @staticmethod
    def _dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
        """
        Apply dynamic range compression to a tensor.

        Args:
            x (torch.Tensor): Input tensor.
            C (float): Compression factor.
            clip_val (float): Clipping value.

        Returns:
            torch.Tensor: Compressed tensor.
        """
        return torch.log(torch.clamp(x, min=clip_val) * C)

    @staticmethod
    def _spectral_normalize_torch(magnitudes):
        """
        Apply spectral normalization to magnitudes.

        Args:
            magnitudes (torch.Tensor): Magnitudes to normalize.

        Returns:
            torch.Tensor: Normalized magnitudes.
        """
        output = HiFiGanDataset._dynamic_range_compression_torch(magnitudes)
        return output

    @staticmethod
    def mel_spectrogram(
        y,
        n_fft,
        num_mels,
        sampling_rate,
        hop_size,
        win_size,
        fmin,
        fmax,
        center=False,
    ):
        """
        Compute the Mel spectrogram of an audio signal.

        Args:
            y (torch.Tensor): Input audio signal.
            n_fft (int): Number of FFT components.
            num_mels (int): Number of Mel bands.
            sampling_rate (int): Sampling rate of the audio.
            hop_size (int): Hop size for STFT.
            win_size (int): Window size for STFT.
            fmin (int): Minimum frequency for Mel filterbank.
            fmax (int): Maximum frequency for Mel filterbank.
            center (bool): Whether to pad the input so that the t-th frame is centered at y[t * hop_length].

        Returns:
            torch.Tensor: Mel spectrogram.
        """
        mel_basis = {}
        hann_window = {}

        if torch.min(y) < -1.0:
            print("min value is ", torch.min(y))
        if torch.max(y) > 1.0:
            print("max value is ", torch.max(y))

        if fmax not in mel_basis:
            mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
            mel_basis[str(fmax) + "_" + str(y.device)] = (
                torch.from_numpy(mel).float().to(y.device)
            )
            hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
        spec = HiFiGanDataset._spectral_normalize_torch(spec)

        return spec

    @staticmethod
    def get_dataset_filelist(
        input_training_file, input_wavs_dir, input_validation_file
    ):
        """
        Generate file lists for training and validation datasets.

        Args:
            input_training_file (str): Path to the file listing the training files.
            input_wavs_dir (str): Directory containing the WAV files.
            input_validation_file (str): Path to the file listing the validation files.

        Returns:
            tuple: A tuple containing two lists: training files and validation files.
        """

        with open(input_training_file, "r", encoding="utf-8") as fi:
            training_files = [
                os.path.join(input_wavs_dir, x.split("|")[0] + ".wav")
                for x in fi.read().split("\n")
                if len(x) > 0
            ]

        with open(input_validation_file, "r", encoding="utf-8") as fi:
            validation_files = [
                os.path.join(input_wavs_dir, x.split("|")[0] + ".wav")
                for x in fi.read().split("\n")
                if len(x) > 0
            ]
        return training_files, validation_files

    def __getitem__(self, index):
        """
        Get an item from the dataset.

        Args:
            index (int): Index of the item to get.

        Returns:
            tuple: A tuple containing the Mel spectrogram, audio, filename, and Mel spectrogram for loss calculation.
        """
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = HiFiGanDataset._load_wav(filename)
            audio = audio / hps.max_wav_value
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    "{} SR doesn't match target {} SR".format(
                        sampling_rate, self.sampling_rate
                    )
                )
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start : audio_start + self.segment_size]
                else:
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

            mel = HiFiGanDataset.mel_spectrogram(
                audio,
                self.n_fft,
                self.num_mels,
                self.sampling_rate,
                self.hop_size,
                self.win_size,
                self.fmin,
                self.fmax,
                center=False,
            )
        else:
            mel = np.load(
                os.path.join(
                    self.base_mels_path,
                    os.path.splitext(os.path.split(filename)[-1])[0] + ".npy",
                )
            ).astype(np.float32)
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start : mel_start + frames_per_seg]
                    audio = audio[
                        :,
                        mel_start
                        * self.hop_size : (mel_start + frames_per_seg)
                        * self.hop_size,
                    ]
                else:
                    mel = torch.nn.functional.pad(
                        mel, (0, frames_per_seg - mel.size(2)), "constant"
                    )
                    audio = torch.nn.functional.pad(
                        audio, (0, self.segment_size - audio.size(1)), "constant"
                    )

        mel_loss = HiFiGanDataset.mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
        )

        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.audio_files)

    @staticmethod
    def dataloader_factory(
        filelist, device, fine_tuning, input_mels_dir, num_gpus, validation=False
    ):
        """
        Factory method to create a DataLoader for the dataset.

        Args:
            filelist (list): List of files to include in the dataset.
            device (torch.device): Device to perform operations on.
            fine_tuning (bool): Whether the dataset is used for fine-tuning.
            input_mels_dir (str): Directory containing Mel spectrograms.
            num_gpus (int): Number of GPUs to use.
            validation (bool): Whether the DataLoader is for validation.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the dataset.
        """
        dataset = HiFiGanDataset(
            filelist,
            hps.segment_size,
            hps.n_fft,
            hps.num_mels,
            hps.hop_size,
            hps.win_size,
            hps.sampling_rate,
            hps.fmin,
            hps.fmax,
            n_cache_reuse=0,
            shuffle=False if hps.num_gpus > 1 else True,
            fmax_loss=hps.fmax_for_loss,
            device=device,
            fine_tuning=fine_tuning,
            base_mels_path=input_mels_dir,
        )

        sampler = DistributedSampler(dataset) if (num_gpus > 1 and validation) else None

        loader = DataLoader(
            dataset,
            num_workers=1 if validation else hps.num_workers,
            shuffle=False,
            sampler=sampler,
            batch_size=1 if validation else hps.batch_size,
            pin_memory=True,
            drop_last=True,
        )

        return loader
