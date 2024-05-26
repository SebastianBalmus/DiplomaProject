import librosa
import numpy as np
from scipy.io import wavfile
from librosa.util import normalize
from hparams.Tacotron2HParams import Tacotron2HParams as hps

MAX_WAV_VALUE = 32768.0
_mel_basis = None


def load_wav(path):
    """
    Load an audio waveform from a file.

    Args:
        path (str): Path to the audio file.

    Returns:
        numpy.ndarray: Loaded audio waveform.
    """
    sr, wav = wavfile.read(path)
    assert sr == hps.sample_rate
    return normalize(wav / MAX_WAV_VALUE) * 0.95


def save_wav(wav, path):
    """
    Save an audio waveform to a file.

    Args:
        wav (numpy.ndarray): Audio waveform to save.
        path (str): Path to save the audio file.
    """
    wav *= MAX_WAV_VALUE
    wavfile.write(path, hps.sample_rate, wav.astype(np.int16))


def spectrogram(y):
    """
    Compute the magnitude spectrogram of an audio waveform.

    Args:
        y (numpy.ndarray): Input audio waveform.

    Returns:
        numpy.ndarray: Magnitude spectrogram.
    """
    D = _stft(y)
    S = _amp_to_db(np.abs(D))
    return S


def inv_spectrogram(S):
    """
    Reconstruct an audio waveform from a magnitude spectrogram.

    Args:
        S (numpy.ndarray): Magnitude spectrogram.

    Returns:
        numpy.ndarray: Reconstructed audio waveform.
    """
    S = _db_to_amp(S)
    return _griffin_lim(S**hps.power)


def melspectrogram(y):
    """
    Compute the mel spectrogram of an audio waveform.

    Args:
        y (numpy.ndarray): Input audio waveform.

    Returns:
        numpy.ndarray: Mel spectrogram.
    """
    D = _stft(y)
    S = _amp_to_db(_linear_to_mel(np.abs(D)))
    return S


def inv_melspectrogram(mel):
    """
    Reconstruct an audio waveform from a mel spectrogram.

    Args:
        mel (numpy.ndarray): Mel spectrogram.

    Returns:
        numpy.ndarray: Reconstructed audio waveform.
    """
    mel = _db_to_amp(mel)
    S = _mel_to_linear(mel)
    return _griffin_lim(S**hps.power)


def _griffin_lim(S):
    """
    Reconstruct an audio waveform from a magnitude spectrogram using Griffin-Lim algorithm.
    Based on https://github.com/librosa/librosa/issues/434

    Args:
        S (numpy.ndarray): Magnitude spectrogram.

    Returns:
        numpy.ndarray: Reconstructed audio waveform.
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles)
    for i in range(hps.gl_iters):
        angles = np.exp(1j * np.angle(_stft(y)))
        y = _istft(S_complex * angles)
    return np.clip(y, a_max=1, a_min=-1)


# Conversions:
def _stft(y):
    """
    Compute the short-time Fourier transform (STFT) of an audio waveform.

    Args:
        y (numpy.ndarray): Input audio waveform.

    Returns:
        numpy.ndarray: Short-time Fourier transform.
    """
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        pad_mode="reflect",
    )


def _istft(y):
    """
    Compute the inverse short-time Fourier transform (iSTFT) of a magnitude spectrogram.

    Args:
        y (numpy.ndarray): Magnitude spectrogram.

    Returns:
        numpy.ndarray: Reconstructed audio waveform.
    """
    _, hop_length, win_length = _stft_parameters()
    return librosa.istft(y, hop_length=hop_length, win_length=win_length)


def _stft_parameters():
    """
    Get the parameters for computing STFT.

    Returns:
        tuple: Parameters for STFT computation (n_fft, hop_length, win_length).
    """
    return (hps.num_freq - 1) * 2, hps.frame_shift, hps.frame_length


def _linear_to_mel(spectrogram):
    """
    Convert a linear-scale spectrogram to a mel-scale spectrogram.

    Args:
        spectrogram (numpy.ndarray): Input linear-scale spectrogram.

    Returns:
        numpy.ndarray: Mel-scale spectrogram.
    """
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _mel_to_linear(spectrogram):
    """
    Convert a mel-scale spectrogram to a linear-scale spectrogram.

    Args:
        spectrogram (numpy.ndarray): Input mel-scale spectrogram.

    Returns:
        numpy.ndarray: Linear-scale spectrogram.
    """
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    inv_mel_basis = np.linalg.pinv(_mel_basis)
    inverse = np.dot(inv_mel_basis, spectrogram)
    inverse = np.maximum(1e-10, inverse)
    return inverse


def _build_mel_basis():
    """
    Build the mel basis matrix for mel-frequency conversion.

    Returns:
        numpy.ndarray: Mel basis matrix.
    """
    n_fft = (hps.num_freq - 1) * 2
    return librosa.filters.mel(
        hps.sample_rate, n_fft, n_mels=hps.num_mels, fmin=hps.fmin, fmax=hps.fmax
    )


def _amp_to_db(x):
    """
    Convert magnitude values to decibels (dB).

    Args:
        x (numpy.ndarray): Input magnitude values.

    Returns:
        numpy.ndarray: Corresponding values in decibels.
    """
    return np.log(np.maximum(1e-5, x))


def _db_to_amp(x):
    """
    Convert decibel values to magnitude values.

    Args:
        x (numpy.ndarray): Input values in decibels.

    Returns:
        numpy.ndarray: Corresponding magnitude values.
    """
    return np.exp(x)
