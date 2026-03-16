"""
Inference module for Speech Emotion Recognition.

Provides a ``Predictor`` class that loads a trained model and scaler,
then predicts the emotion of a single audio file with confidence
scores and visualisation helpers (waveform, MFCC heatmap).
"""

import os
import logging
from typing import Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import librosa
import librosa.display
import joblib
import keras

from src import config
from src.feature_extraction import (
    extract_features,
    extract_features_3d,
    extract_mfcc_for_visualization,
)

# Use non-interactive backend so plots can be generated headlessly
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


class Predictor:
    """
    End-to-end predictor for Speech Emotion Recognition.

    Loads a trained Keras model and its corresponding scikit-learn
    ``StandardScaler``, then performs inference on a single .wav file.

    Attributes:
        model: Loaded Keras model.
        scaler: Loaded StandardScaler.
        model_type: ``"mlp"`` or ``"lstm"``.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        model_type: str = "lstm",
    ) -> None:
        """
        Initialise the Predictor.

        Args:
            model_path: Path to the saved ``.keras`` model file.
                        Defaults to ``models/{model_type}_model.keras``.
            scaler_path: Path to the saved ``.pkl`` scaler file.
                         Defaults based on *model_type*.
            model_type: ``"mlp"`` or ``"lstm"``.

        Raises:
            FileNotFoundError: If the model or scaler files are missing.
        """
        self.model_type = model_type.lower()

        # Resolve default paths
        if model_path is None:
            model_path = os.path.join(
                config.MODEL_SAVE_PATH, f"{self.model_type}_model.keras"
            )
        if scaler_path is None:
            scaler_key = "scaler_flat.pkl" if self.model_type == "mlp" else "scaler_3d.pkl"
            scaler_path = os.path.join(config.MODEL_SAVE_PATH, scaler_key)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.isfile(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        self.model: keras.Model = keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)

        logger.info(
            "Predictor ready — model: %s, scaler: %s", model_path, scaler_path
        )

    # ────────────────────────────────────────────
    # Core prediction
    # ────────────────────────────────────────────
    def predict(self, audio_file_path: str) -> dict[str, Any]:
        """
        Predict the emotion of a single audio file.

        Args:
            audio_file_path: Path to a ``.wav`` file.

        Returns:
            A dictionary containing:
                - ``predicted_emotion`` (str): Top-1 label.
                - ``confidence`` (float): Probability of the top-1 label.
                - ``all_scores`` (dict[str, float]): Probabilities for
                  every emotion class.
                - ``waveform_plot`` (matplotlib.figure.Figure): Waveform.
                - ``mfcc_plot`` (matplotlib.figure.Figure): MFCC heatmap.

        Raises:
            FileNotFoundError: If *audio_file_path* does not exist.
            RuntimeError: If feature extraction fails.
        """
        if not os.path.isfile(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

        # Extract features based on model type
        if self.model_type == "mlp":
            features = extract_features(audio_file_path)
            if features is None:
                raise RuntimeError(
                    f"Feature extraction failed for {audio_file_path}"
                )
            # Scale and reshape to (1, 530)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
        else:
            features_3d = extract_features_3d(audio_file_path)
            if features_3d is None:
                raise RuntimeError(
                    f"Feature extraction failed for {audio_file_path}"
                )
            # Scale: reshape → scale → reshape back
            n_steps, n_feat = features_3d.shape
            features_scaled = self.scaler.transform(
                features_3d.reshape(-1, n_feat)
            ).reshape(1, n_steps, n_feat)

        # Predict
        probabilities = self.model.predict(features_scaled, verbose=0)[0]
        predicted_idx = int(np.argmax(probabilities))
        predicted_emotion = config.EMOTIONS[predicted_idx + 1]
        confidence = float(probabilities[predicted_idx])

        # Build score dictionary
        all_scores: dict[str, float] = {
            config.EMOTIONS[i + 1]: round(float(probabilities[i]), 4)
            for i in range(config.NUM_CLASSES)
        }

        # Visualisations
        waveform_fig = self._plot_waveform(audio_file_path)
        mfcc_fig = self._plot_mfcc(audio_file_path)

        result = {
            "predicted_emotion": predicted_emotion,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "waveform_plot": waveform_fig,
            "mfcc_plot": mfcc_fig,
        }

        logger.info(
            "Prediction: %s (%.2f%% confidence)", predicted_emotion, confidence * 100
        )
        return result

    # ────────────────────────────────────────────
    # Visualisations
    # ────────────────────────────────────────────
    def _plot_waveform(self, audio_file_path: str) -> plt.Figure:
        """
        Generate a waveform plot for the given audio file.

        Args:
            audio_file_path: Path to the .wav file.

        Returns:
            A ``matplotlib.figure.Figure`` object.
        """
        y, sr = librosa.load(
            audio_file_path, sr=config.SAMPLE_RATE, duration=config.DURATION
        )

        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#4A90D9")
        ax.set_title("Waveform", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        fig.tight_layout()
        return fig

    def _plot_mfcc(self, audio_file_path: str) -> plt.Figure:
        """
        Generate an MFCC heatmap for the given audio file.

        Args:
            audio_file_path: Path to the .wav file.

        Returns:
            A ``matplotlib.figure.Figure`` object.
        """
        mfcc = extract_mfcc_for_visualization(audio_file_path)

        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            mfcc,
            x_axis="time",
            sr=config.SAMPLE_RATE,
            hop_length=config.HOP_LENGTH,
            ax=ax,
            cmap="magma",
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        ax.set_title("MFCC Heatmap", fontsize=14, fontweight="bold")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MFCC Coefficient")
        fig.tight_layout()
        return fig


# ────────────────────────────────────────────────
# CLI entry point
# ────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.predict <path_to_wav_file> [model_type]")
        sys.exit(1)

    wav_path = sys.argv[1]
    m_type = sys.argv[2] if len(sys.argv) > 2 else "lstm"

    predictor = Predictor(model_type=m_type)
    result = predictor.predict(wav_path)

    print(f"\n🎯 Predicted Emotion: {result['predicted_emotion']}")
    print(f"   Confidence:        {result['confidence']:.2%}")
    print("\n📊 All Scores:")
    for emotion, score in sorted(
        result["all_scores"].items(), key=lambda x: x[1], reverse=True
    ):
        bar = "█" * int(score * 40)
        print(f"   {emotion:<12s} {score:.4f}  {bar}")
