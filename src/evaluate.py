"""
Evaluation module for Speech Emotion Recognition.

Loads saved models and test data, then generates:
    a. Overall accuracy, precision, recall, F1 (macro & weighted).
    b. Per-class classification report.
    c. Normalised confusion matrix heatmap (seaborn).
    d. ROC-AUC curves (one-vs-rest, per emotion).
    e. Bar chart comparing MLP vs LSTM accuracy per emotion.
"""

import os
import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import keras

from src import config

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Load helpers
# ────────────────────────────────────────────────────────────
def _load_test_data() -> dict:
    """
    Load saved test data from ``features/test_data.npz``.

    Returns:
        Dictionary with keys ``X_flat_test``, ``X_3d_test``,
        ``y_test``, and ``y_test_cat``.

    Raises:
        FileNotFoundError: If the test data file is missing.
    """
    path = os.path.join(config.FEATURES_PATH, "test_data.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Test data not found at {path}. Run train.py first."
        )

    data = np.load(path, allow_pickle=True)
    logger.info("Loaded test data from %s", path)
    return dict(data)


def _load_model(name: str) -> keras.Model:
    """
    Load a saved Keras model by name.

    Args:
        name: One of ``"mlp"`` or ``"lstm"``.

    Returns:
        Loaded ``keras.Model``.

    Raises:
        FileNotFoundError: If the model file is missing.
    """
    path = os.path.join(config.MODEL_SAVE_PATH, f"{name}_model.keras")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}. Run train.py first.")

    model = keras.models.load_model(path)
    logger.info("Loaded model from %s", path)
    return model


# ────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict:
    """
    Compute and log overall classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        model_name: Label string for logging.

    Returns:
        Dictionary of computed metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

    logger.info("=" * 50)
    logger.info("[%s] Evaluation Metrics", model_name.upper())
    logger.info("  Accuracy          : %.4f", acc)
    logger.info("  Precision (macro) : %.4f", prec_macro)
    logger.info("  Recall    (macro) : %.4f", rec_macro)
    logger.info("  F1 Score  (macro) : %.4f", f1_macro)
    logger.info("  F1 Score  (weighted): %.4f", f1_weighted)
    logger.info("=" * 50)

    return metrics


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> str:
    """
    Print a per-class classification report.

    Args:
        y_true: Ground-truth labels (integers).
        y_pred: Predicted labels (integers).
        model_name: Label for logging.

    Returns:
        The report as a formatted string.
    """
    emotion_names = [config.EMOTIONS[i + 1] for i in range(config.NUM_CLASSES)]
    report = classification_report(
        y_true, y_pred, target_names=emotion_names, zero_division=0
    )
    logger.info("[%s] Classification Report:\n%s", model_name.upper(), report)
    return report


# ────────────────────────────────────────────────────────────
# Confusion Matrix
# ────────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_dir: str = config.PLOTS_PATH,
) -> None:
    """
    Plot and save a normalised confusion matrix heatmap.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        model_name: Label for the title and filename.
        save_dir: Directory to save the figure.

    Returns:
        None
    """
    emotion_names = [config.EMOTIONS[i + 1] for i in range(config.NUM_CLASSES)]

    cm = confusion_matrix(y_true, y_pred)
    # Normalise each row (true label) to sum to 1
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=emotion_names,
        yticklabels=emotion_names,
        linewidths=0.5,
    )
    plt.title(f"{model_name.upper()} — Normalised Confusion Matrix", fontsize=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", path)


# ────────────────────────────────────────────────────────────
# ROC-AUC Curves
# ────────────────────────────────────────────────────────────
def plot_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    save_dir: str = config.PLOTS_PATH,
) -> None:
    """
    Plot one-vs-rest ROC curves and per-class AUC.

    Args:
        y_true: Ground-truth integer labels.
        y_prob: Predicted probability matrix ``(n_samples, num_classes)``.
        model_name: Label for the title and filename.
        save_dir: Directory to save the figure.

    Returns:
        None
    """
    emotion_names = [config.EMOTIONS[i + 1] for i in range(config.NUM_CLASSES)]
    n_classes = config.NUM_CLASSES

    # Binarise true labels for OVR
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    plt.figure(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i, (emotion, color) in enumerate(zip(emotion_names, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{emotion} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"{model_name.upper()} — ROC Curves (One-vs-Rest)", fontsize=15)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, f"{model_name}_roc_auc.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("ROC-AUC curves saved to %s", path)


# ────────────────────────────────────────────────────────────
# Comparison bar chart
# ────────────────────────────────────────────────────────────
def plot_model_comparison(
    y_true: np.ndarray,
    y_pred_mlp: np.ndarray,
    y_pred_lstm: np.ndarray,
    save_dir: str = config.PLOTS_PATH,
) -> None:
    """
    Side-by-side bar chart of per-emotion accuracy: MLP vs LSTM.

    Args:
        y_true: Ground-truth integer labels.
        y_pred_mlp: MLP predictions.
        y_pred_lstm: LSTM predictions.
        save_dir: Directory to save the figure.

    Returns:
        None
    """
    emotion_names = [config.EMOTIONS[i + 1] for i in range(config.NUM_CLASSES)]
    mlp_acc_per_class: list[float] = []
    lstm_acc_per_class: list[float] = []

    for class_idx in range(config.NUM_CLASSES):
        mask = y_true == class_idx
        if mask.sum() == 0:
            mlp_acc_per_class.append(0.0)
            lstm_acc_per_class.append(0.0)
        else:
            mlp_acc_per_class.append(
                accuracy_score(y_true[mask], y_pred_mlp[mask])
            )
            lstm_acc_per_class.append(
                accuracy_score(y_true[mask], y_pred_lstm[mask])
            )

    x = np.arange(len(emotion_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_mlp = ax.bar(x - width / 2, mlp_acc_per_class, width, label="MLP", color="#6C8EBF")
    bars_lstm = ax.bar(x + width / 2, lstm_acc_per_class, width, label="LSTM", color="#82B366")

    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Per-Emotion Accuracy: MLP vs LSTM", fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_names, rotation=30, ha="right")
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    # Label bars
    for bar in bars_mlp:
        ax.annotate(
            f"{bar.get_height():.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8,
        )
    for bar in bars_lstm:
        ax.annotate(
            f"{bar.get_height():.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    path = os.path.join(save_dir, "mlp_vs_lstm_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Model comparison chart saved to %s", path)


# ────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ────────────────────────────────────────────────────────────
def evaluate() -> None:
    """
    Run full evaluation for both MLP and LSTM models.

    Loads saved models and test data, then generates all plots and
    logs all metrics.

    Returns:
        None
    """
    # Load test data
    test_data = _load_test_data()
    X_flat_test = test_data["X_flat_test"]
    X_3d_test = test_data["X_3d_test"]
    y_test = test_data["y_test"]

    # Load models
    mlp = _load_model("mlp")
    lstm = _load_model("lstm")

    # ── MLP evaluation ──
    mlp_probs = mlp.predict(X_flat_test, verbose=0)
    mlp_preds = np.argmax(mlp_probs, axis=1)

    compute_metrics(y_test, mlp_preds, "mlp")
    print_classification_report(y_test, mlp_preds, "mlp")
    plot_confusion_matrix(y_test, mlp_preds, "mlp")
    plot_roc_auc(y_test, mlp_probs, "mlp")

    # ── LSTM evaluation ──
    lstm_probs = lstm.predict(X_3d_test, verbose=0)
    lstm_preds = np.argmax(lstm_probs, axis=1)

    compute_metrics(y_test, lstm_preds, "lstm")
    print_classification_report(y_test, lstm_preds, "lstm")
    plot_confusion_matrix(y_test, lstm_preds, "lstm")
    plot_roc_auc(y_test, lstm_probs, "lstm")

    # ── Comparison chart ──
    plot_model_comparison(y_test, mlp_preds, lstm_preds)

    logger.info("Evaluation complete. All plots saved to %s", config.PLOTS_PATH)


# ────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    evaluate()
