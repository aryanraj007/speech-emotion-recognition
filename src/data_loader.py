"""
Data loader for the RAVDESS dataset.

Parses the RAVDESS filename convention to extract metadata such as
emotion label, actor ID, and gender. Returns a structured pandas
DataFrame suitable for downstream feature extraction and training.

RAVDESS filename format (7 identifiers separated by hyphens):
    Modality-VocalChannel-Emotion-EmotionalIntensity-Statement-Repetition-Actor.wav
    e.g.  03-01-05-01-01-01-12.wav
          → Emotion code = 5 → "Angry", Actor = 12 → Female (even)
"""

import os
import glob
import logging
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import config

# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────
logger = logging.getLogger(__name__)


def load_ravdess_dataset(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Scan the RAVDESS data directory and build a metadata DataFrame.

    The function traverses all sub-directories under *data_path* looking
    for ``.wav`` files that follow the RAVDESS naming convention.

    Args:
        data_path: Root folder containing the RAVDESS .wav files.
                   Defaults to ``config.DATA_PATH`` when *None*.

    Returns:
        A ``pd.DataFrame`` with columns:
        ``file_path``, ``emotion_code``, ``emotion_label``, ``actor``, ``gender``.

    Raises:
        FileNotFoundError: If *data_path* does not exist.
        ValueError: If no .wav files are found.
    """
    if data_path is None:
        data_path = config.DATA_PATH

    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            f"Data directory not found: {data_path}. "
            "Please download the RAVDESS dataset and place the .wav files "
            "inside the data/raw/ folder."
        )

    # Collect all .wav files recursively
    wav_files: list[str] = sorted(
        glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    )

    if not wav_files:
        raise ValueError(
            f"No .wav files found in {data_path}. "
            "Ensure RAVDESS audio files are present."
        )

    logger.info("Found %d .wav files in %s", len(wav_files), data_path)

    records: list[dict] = []
    for fpath in wav_files:
        try:
            basename = os.path.splitext(os.path.basename(fpath))[0]
            parts = basename.split("-")

            if len(parts) != 7:
                logger.warning("Skipping non-RAVDESS file: %s", fpath)
                continue

            emotion_code = int(parts[2])
            actor_id = int(parts[6])
            # In RAVDESS, odd actor IDs are male, even are female
            gender = "Female" if actor_id % 2 == 0 else "Male"

            records.append(
                {
                    "file_path": fpath,
                    "emotion_code": emotion_code,
                    "emotion_label": config.EMOTIONS.get(emotion_code, "Unknown"),
                    "actor": actor_id,
                    "gender": gender,
                }
            )
        except (ValueError, IndexError) as exc:
            logger.warning("Could not parse filename %s: %s", fpath, exc)

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d valid samples across %d emotion classes.",
        len(df),
        df["emotion_label"].nunique(),
    )
    return df


def get_class_distribution(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Print and plot the class distribution of emotions in the dataset.

    Generates a seaborn bar chart showing the number of samples per
    emotion label and saves the figure if *save_path* is provided.

    Args:
        df: DataFrame returned by :func:`load_ravdess_dataset`.
        save_path: Optional path to save the plot image.
                   Defaults to ``config.PLOTS_PATH / class_distribution.png``.

    Returns:
        None
    """
    if save_path is None:
        save_path = os.path.join(config.PLOTS_PATH, "class_distribution.png")

    counts = df["emotion_label"].value_counts().sort_index()
    logger.info("Class distribution:\n%s", counts.to_string())

    # ── Plot ──
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(
        data=df,
        x="emotion_label",
        order=sorted(df["emotion_label"].unique()),
        hue="emotion_label",
        palette="viridis",
        legend=False,
    )
    ax.set_title("RAVDESS Emotion Class Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Emotion", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)

    # Add count labels on top of each bar
    for patch in ax.patches:
        ax.annotate(
            f"{int(patch.get_height())}",
            (patch.get_x() + patch.get_width() / 2.0, patch.get_height()),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Class distribution plot saved to %s", save_path)


def get_gender_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cross-tabulation of emotion labels by gender.

    Args:
        df: DataFrame returned by :func:`load_ravdess_dataset`.

    Returns:
        A ``pd.DataFrame`` cross-tab with emotions as rows and
        genders as columns.
    """
    cross_tab = pd.crosstab(df["emotion_label"], df["gender"])
    logger.info("Gender × Emotion distribution:\n%s", cross_tab.to_string())
    return cross_tab


# ─────────────────────────────────────────────
# Quick test when run as a script
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    dataset = load_ravdess_dataset()
    print(dataset.head(10))
    print(f"\nTotal samples: {len(dataset)}")
    get_class_distribution(dataset)
