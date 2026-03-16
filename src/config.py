"""
Configuration module for Speech Emotion Recognition system.

Contains all hyperparameters, file paths, and constants used
throughout the project. Centralizes configuration to ensure
consistency and easy experimentation.
"""

import os

# ─────────────────────────────────────────────
# Directory Paths
# ─────────────────────────────────────────────
# Root directory of the project (parent of src/)
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path where raw RAVDESS .wav files are stored
DATA_PATH: str = os.path.join(PROJECT_ROOT, "data", "raw")

# Path where trained models and artefacts are saved
MODEL_SAVE_PATH: str = os.path.join(PROJECT_ROOT, "models")

# Path for extracted feature caches
FEATURES_PATH: str = os.path.join(PROJECT_ROOT, "features")

# Path for output figures / plots
PLOTS_PATH: str = os.path.join(PROJECT_ROOT, "plots")

# Ensure output directories exist
for _dir in [MODEL_SAVE_PATH, FEATURES_PATH, PLOTS_PATH]:
    os.makedirs(_dir, exist_ok=True)

# ─────────────────────────────────────────────
# Audio Processing Parameters
# ─────────────────────────────────────────────
SAMPLE_RATE: int = 22050           # Sampling rate for librosa
DURATION: int = 3                  # Fixed duration in seconds (pad/trim)
N_MFCC: int = 40                   # Number of MFCC coefficients
N_MELS: int = 128                  # Number of Mel filter banks
HOP_LENGTH: int = 512              # Hop length for STFT window

# ─────────────────────────────────────────────
# Emotion Label Mapping  (RAVDESS encoding)
# ─────────────────────────────────────────────
# RAVDESS filename convention:  Modality-VocalChannel-Emotion-...
# The 3rd position encodes the emotion as an integer 01–08.
EMOTIONS: dict[int, str] = {
    1: "Neutral",
    2: "Calm",
    3: "Happy",
    4: "Sad",
    5: "Angry",
    6: "Fearful",
    7: "Disgusted",
    8: "Surprised",
}

# Number of emotion classes
NUM_CLASSES: int = len(EMOTIONS)

# Emoji mapping for the Gradio UI
EMOTION_EMOJIS: dict[str, str] = {
    "Neutral": "😐",
    "Calm": "😌",
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Fearful": "😨",
    "Disgusted": "🤢",
    "Surprised": "😲",
}

# ─────────────────────────────────────────────
# Train / Test Split
# ─────────────────────────────────────────────
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ─────────────────────────────────────────────
# Training Hyperparameters
# ─────────────────────────────────────────────
EPOCHS: int = 80
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001

# ─────────────────────────────────────────────
# Feature Dimensions  (computed from the above)
# ─────────────────────────────────────────────
# MFCC mean+std            : 40*2 = 80
# Delta MFCC mean+std      : 40*2 = 80
# Delta-Delta MFCC mean+std: 40*2 = 80
# Mel Spectrogram mean+std : 128*2 = 256
# Chroma STFT mean+std     : 12*2 = 24
# ZCR mean+std             : 1*2 = 2
# RMS Energy mean+std      : 1*2 = 2
# Spectral Centroid mean+std: 1*2 = 2
# Spectral Bandwidth mean+std: 1*2 = 2
# Spectral Rolloff mean+std : 1*2 = 2
# ─────────────────────────────── Total : 530
FEATURE_DIM: int = 530

# Time steps for the LSTM input (DURATION * SAMPLE_RATE / HOP_LENGTH)
# librosa produces ceil(n_samples / hop_length) + 1 frames
LSTM_TIME_STEPS: int = int((DURATION * SAMPLE_RATE) / HOP_LENGTH) + 1
