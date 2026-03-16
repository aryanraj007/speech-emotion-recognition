"""
Gradio-based web UI for Speech Emotion Recognition.

Provides an interactive interface where users can either upload a
``.wav`` file or record audio via microphone. The app displays:
    - The predicted emotion with an emoji label.
    - A horizontal bar chart of confidence scores for all 8 emotions.
    - Waveform visualisation.
    - MFCC heatmap.
"""

import os
import sys
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gradio as gr

# Ensure the project root is on sys.path so `src` can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.predict import Predictor

# Non-interactive backend for headless environments
matplotlib.use("Agg")

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Global predictor (loaded once on app start)
# ────────────────────────────────────────────────────────────
def _get_predictor() -> Predictor:
    """
    Initialise and return a Predictor, falling back to MLP if
    the LSTM model is not available.

    Returns:
        A ready-to-use ``Predictor`` instance.
    """
    try:
        return Predictor(model_type="lstm")
    except FileNotFoundError:
        logger.warning("LSTM model not found, falling back to MLP.")
        try:
            return Predictor(model_type="mlp")
        except FileNotFoundError:
            logger.error(
                "No trained model found. Please run `python -m src.train` first."
            )
            raise


# ────────────────────────────────────────────────────────────
# Prediction callback
# ────────────────────────────────────────────────────────────
def predict_emotion(audio_path: str):
    """
    Gradio callback: run inference and return UI components.

    Args:
        audio_path: Path to the uploaded / recorded .wav file
                    (provided by Gradio's Audio component).

    Returns:
        Tuple of:
            - emotion_text (str): Emoji + emotion label string.
            - bar_chart (Figure): Confidence bar chart.
            - waveform (Figure):  Waveform plot.
            - mfcc (Figure):      MFCC heatmap.
    """
    if audio_path is None:
        return "⚠️ Please upload or record an audio file.", None, None, None

    try:
        predictor = _get_predictor()
        result = predictor.predict(audio_path)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        return f"❌ Error: {exc}", None, None, None

    # ── Emoji + label ──
    emoji = config.EMOTION_EMOJIS.get(result["predicted_emotion"], "🎤")
    emotion_text = (
        f"## {emoji} {result['predicted_emotion']}\n"
        f"**Confidence:** {result['confidence']:.1%}"
    )

    # ── Bar chart of all scores ──
    bar_fig = _make_bar_chart(result["all_scores"], result["predicted_emotion"])

    return emotion_text, bar_fig, result["waveform_plot"], result["mfcc_plot"]


def _make_bar_chart(
    scores: dict[str, float], predicted: str
) -> plt.Figure:
    """
    Create a horizontal bar chart of emotion confidence scores.

    Args:
        scores: Mapping of emotion names to probabilities.
        predicted: The top-1 predicted emotion (highlighted).

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    # Sort by score descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    # Colour palette: highlight the predicted emotion
    colours = [
        "#4CAF50" if label == predicted else "#78909C" for label in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colours, edgecolor="white", height=0.6)

    # Add percentage labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_title("Emotion Confidence Scores", fontsize=14, fontweight="bold")
    ax.invert_yaxis()  # Highest score on top
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────
# Gradio Interface
# ────────────────────────────────────────────────────────────
def build_app() -> gr.Blocks:
    """
    Construct the Gradio Blocks layout.

    Returns:
        A ``gr.Blocks`` app ready to be launched.
    """
    with gr.Blocks(
        title="🎙️ Speech Emotion Recognition",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎙️ Speech Emotion Recognition

            Upload a `.wav` file **or** record from your microphone and
            the model will detect the emotion in your speech.

            > **Model:** LSTM trained on the RAVDESS dataset \
            > (Ryerson Audio-Visual Database of Emotional Speech and Song). \
            > **Features:** 40-dim MFCC sequences processed by a two-layer \
            > LSTM with dropout and batch normalisation. \
            > **Emotions:** Neutral · Calm · Happy · Sad · Angry · Fearful · \
            > Disgusted · Surprised
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    label="🎤 Upload or Record Audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                submit_btn = gr.Button(
                    "🔍 Analyse Emotion", variant="primary", size="lg"
                )

            with gr.Column(scale=2):
                emotion_output = gr.Markdown(label="Detected Emotion")
                confidence_plot = gr.Plot(label="Confidence Scores")

        with gr.Row():
            waveform_plot = gr.Plot(label="Waveform")
            mfcc_plot = gr.Plot(label="MFCC Heatmap")

        # Wire up the callback
        submit_btn.click(
            fn=predict_emotion,
            inputs=[audio_input],
            outputs=[emotion_output, confidence_plot, waveform_plot, mfcc_plot],
        )

        # Example files (if any exist in data/raw/)
        example_dir = config.DATA_PATH
        if os.path.isdir(example_dir):
            import glob

            examples = sorted(glob.glob(os.path.join(example_dir, "**", "*.wav"), recursive=True))[:5]
            if examples:
                gr.Examples(
                    examples=[[ex] for ex in examples],
                    inputs=[audio_input],
                    label="📁 Example Files",
                )

    return demo


# ────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
