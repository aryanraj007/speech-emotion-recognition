"""
Feature extraction module for Speech Emotion Recognition.

Provides two extraction functions:
    1. ``extract_features`` — Aggregated 1-D feature vector (530 dims)
       combining statistics of MFCC, delta-MFCC, mel-spectrogram,
       chroma, ZCR, RMS, spectral centroid/bandwidth/rolloff.
    2. ``extract_features_3d`` — 2-D array of raw MFCC frames
       (time_steps × n_mfcc) ready for LSTM consumption.
"""

import logging
from typing import Optional

import numpy as np
import librosa

from src import config

logger = logging.getLogger(__name__)


def _load_audio(
    file_path: str,
    sr: int = config.SAMPLE_RATE,
    duration: int = config.DURATION,
) -> np.ndarray:
    """
    Load a .wav file, trim silence, then pad or truncate to a fixed duration.

    Args:
        file_path: Absolute or relative path to the .wav file.
        sr: Target sampling rate.
        duration: Desired duration in seconds.

    Returns:
        1-D numpy array of audio samples with length ``sr * duration``.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        RuntimeError: If the file is corrupt or unreadable.
    """
    try:
        # Load audio at the target sample rate
        y, _ = librosa.load(file_path, sr=sr, duration=duration)
    except FileNotFoundError:
        logger.error("Audio file not found: %s", file_path)
        raise
    except Exception as exc:
        logger.error("Failed to load audio %s: %s", file_path, exc)
        raise RuntimeError(f"Corrupt or unreadable audio: {file_path}") from exc

    # Fixed number of samples
    target_length = sr * duration

    if len(y) < target_length:
        # Pad with zeros (silence) at the end
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        # Truncate to the exact length
        y = y[:target_length]

    return y


def extract_features(
    file_path: str,
    sr: int = config.SAMPLE_RATE,
    duration: int = config.DURATION,
    n_mfcc: int = config.N_MFCC,
    n_mels: int = config.N_MELS,
    hop_length: int = config.HOP_LENGTH,
) -> Optional[np.ndarray]:
    """
    Extract an aggregated 1-D feature vector from a single audio file.

    The following features are computed, and for each the **mean** and
    **standard deviation** across time frames are taken:

    +-----+--------------------+-------+
    | #   | Feature            | Dims  |
    +-----+--------------------+-------+
    | a   | MFCC               | 80    |
    | b   | Δ MFCC             | 80    |
    | c   | ΔΔ MFCC            | 80    |
    | d   | Mel Spectrogram    | 256   |
    | e   | Chroma STFT        | 24    |
    | f   | Zero Crossing Rate | 2     |
    | g   | RMS Energy         | 2     |
    | h   | Spectral Centroid  | 2     |
    | i   | Spectral Bandwidth | 2     |
    | j   | Spectral Rolloff   | 2     |
    +-----+--------------------+-------+
    | Total                    | 530   |
    +-----+--------------------+-------+

    Args:
        file_path: Path to the .wav file.
        sr: Sampling rate.
        duration: Fixed duration in seconds.
        n_mfcc: Number of MFCC coefficients.
        n_mels: Number of mel filter banks.
        hop_length: Hop length for STFT.

    Returns:
        A 1-D numpy array of shape ``(530,)`` or ``None`` if extraction
        fails for the given file.
    """
    try:
        y = _load_audio(file_path, sr=sr, duration=duration)

        # ── Compute intermediate spectral representations ──
        stft = np.abs(librosa.stft(y, hop_length=hop_length))

        # (a) MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

        # (b) Delta MFCC (1st order)
        delta_mfcc = librosa.feature.delta(mfcc)

        # (c) Delta-Delta MFCC (2nd order)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        # (d) Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, hop_length=hop_length
        )

        # (e) Chroma STFT (12 bins)
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr, hop_length=hop_length)

        # (f) Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)

        # (g) RMS Energy
        rms = librosa.feature.rms(y=y, hop_length=hop_length)

        # (h) Spectral Centroid
        spec_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=hop_length
        )

        # (i) Spectral Bandwidth
        spec_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, hop_length=hop_length
        )

        # (j) Spectral Rolloff
        spec_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, hop_length=hop_length
        )

        # ── Aggregate: mean and std across the time axis ──
        def _stat(feat: np.ndarray) -> np.ndarray:
            """Return concatenated mean and std across columns (time axis)."""
            return np.concatenate([feat.mean(axis=1), feat.std(axis=1)])

        features = np.concatenate(
            [
                _stat(mfcc),          # 80
                _stat(delta_mfcc),    # 80
                _stat(delta2_mfcc),   # 80
                _stat(mel_spec),      # 256
                _stat(chroma),        # 24
                _stat(zcr),           # 2
                _stat(rms),           # 2
                _stat(spec_centroid), # 2
                _stat(spec_bandwidth),# 2
                _stat(spec_rolloff),  # 2
            ]
        )

        assert features.shape == (config.FEATURE_DIM,), (
            f"Expected feature dim {config.FEATURE_DIM}, got {features.shape[0]}"
        )

        return features

    except Exception as exc:
        logger.error("Feature extraction failed for %s: %s", file_path, exc)
        return None


def extract_features_3d(
    file_path: str,
    sr: int = config.SAMPLE_RATE,
    duration: int = config.DURATION,
    n_mfcc: int = config.N_MFCC,
    hop_length: int = config.HOP_LENGTH,
) -> Optional[np.ndarray]:
    """
    Extract a 2-D MFCC matrix for LSTM input.

    Instead of aggregating over time, this function returns the raw
    MFCC frames transposed to ``(time_steps, n_mfcc)`` so each time
    step is a feature vector the LSTM can process sequentially.

    Args:
        file_path: Path to the .wav file.
        sr: Sampling rate.
        duration: Fixed duration in seconds.
        n_mfcc: Number of MFCC coefficients.
        hop_length: Hop length for STFT.

    Returns:
        A 2-D numpy array of shape ``(time_steps, n_mfcc)`` or ``None``
        if extraction fails.
    """
    try:
        y = _load_audio(file_path, sr=sr, duration=duration)

        # Compute MFCC → shape (n_mfcc, time_steps)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

        # Transpose to (time_steps, n_mfcc) for LSTM
        mfcc_t = mfcc.T

        # Pad or trim to LSTM_TIME_STEPS for consistency
        target_steps = config.LSTM_TIME_STEPS
        if mfcc_t.shape[0] < target_steps:
            pad_width = target_steps - mfcc_t.shape[0]
            mfcc_t = np.pad(mfcc_t, ((0, pad_width), (0, 0)), mode="constant")
        else:
            mfcc_t = mfcc_t[:target_steps, :]

        return mfcc_t

    except Exception as exc:
        logger.error("3D feature extraction failed for %s: %s", file_path, exc)
        return None


def extract_mfcc_for_visualization(
    file_path: str,
    sr: int = config.SAMPLE_RATE,
    duration: int = config.DURATION,
    n_mfcc: int = config.N_MFCC,
    hop_length: int = config.HOP_LENGTH,
) -> Optional[np.ndarray]:
    """
    Extract raw MFCC matrix (n_mfcc, time) for heatmap visualization.

    Unlike :func:`extract_features_3d`, this keeps the original
    orientation so it can be displayed directly with ``librosa.display``.

    Args:
        file_path: Path to the .wav file.
        sr: Sampling rate.
        duration: Fixed duration in seconds.
        n_mfcc: Number of MFCC coefficients.
        hop_length: Hop length for STFT.

    Returns:
        2-D numpy array of shape ``(n_mfcc, time_steps)`` or ``None``.
    """
    try:
        y = _load_audio(file_path, sr=sr, duration=duration)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        return mfcc
    except Exception as exc:
        logger.error("MFCC visualization extraction failed for %s: %s", file_path, exc)
        return None
