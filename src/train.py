"""
Training pipeline for Speech Emotion Recognition.

Orchestrates:
    1. Dataset loading and feature extraction (with tqdm progress).
    2. Stratified train/test split.
    3. Feature normalisation (StandardScaler, persisted to disk).
    4. One-hot label encoding.
    5. Training of **both** MLP and LSTM models with callbacks.
    6. Training-history visualisation (accuracy & loss curves).
    7. Saving trained models as ``.keras`` files.
"""

import os
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
import joblib

from src import config
from src.data_loader import load_ravdess_dataset, get_class_distribution
from src.feature_extraction import extract_features, extract_features_3d
from src.model import build_mlp_model, build_lstm_model, compile_model

# Suppress noisy TF warnings during training
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────
def set_seeds(seed: int = config.RANDOM_STATE) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow.

    Args:
        seed: Integer seed for reproducibility.

    Returns:
        None
    """
    import random
    import keras

    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seeds set to %d.", seed)


# ────────────────────────────────────────────────────────────
# Feature extraction helpers
# ────────────────────────────────────────────────────────────
def extract_all_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract MLP (1-D) and LSTM (2-D) features for every sample in *df*.

    Args:
        df: DataFrame from :func:`data_loader.load_ravdess_dataset`.

    Returns:
        A tuple ``(X_flat, X_3d, y)`` where
        - ``X_flat``: shape ``(n_samples, 530)``
        - ``X_3d``:   shape ``(n_samples, time_steps, n_mfcc)``
        - ``y``:      integer emotion codes,  shape ``(n_samples,)``
    """
    X_flat_list: list[np.ndarray] = []
    X_3d_list: list[np.ndarray] = []
    y_list: list[int] = []

    logger.info("Extracting features for %d audio files …", len(df))

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Feature extraction"):
        feat_flat = extract_features(row["file_path"])
        feat_3d = extract_features_3d(row["file_path"])

        if feat_flat is not None and feat_3d is not None:
            X_flat_list.append(feat_flat)
            X_3d_list.append(feat_3d)
            y_list.append(row["emotion_code"] - 1)  # zero-indexed
        else:
            logger.warning("Skipping file (extraction failed): %s", row["file_path"])

    X_flat = np.array(X_flat_list, dtype=np.float32)
    X_3d = np.array(X_3d_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    logger.info(
        "Extracted features — X_flat: %s, X_3d: %s, y: %s",
        X_flat.shape,
        X_3d.shape,
        y.shape,
    )
    return X_flat, X_3d, y


# ────────────────────────────────────────────────────────────
# Callbacks
# ────────────────────────────────────────────────────────────
def get_callbacks(model_name: str) -> list:
    """
    Return a list of Keras callbacks for training.

    Args:
        model_name: Either ``"mlp"`` or ``"lstm"``, used in
                    checkpoint file naming.

    Returns:
        List of callback instances.
    """
    checkpoint_path = os.path.join(
        config.MODEL_SAVE_PATH, f"best_{model_name}.keras"
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return callbacks


# ────────────────────────────────────────────────────────────
# Plotting helpers
# ────────────────────────────────────────────────────────────
def plot_training_history(
    history,
    model_name: str,
    save_dir: str = config.PLOTS_PATH,
) -> None:
    """
    Plot training vs. validation accuracy and loss curves.

    Args:
        history: ``keras.callbacks.History`` object from ``model.fit()``.
        model_name: Label used in titles and filenames.
        save_dir: Directory to save figure images.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{model_name.upper()} — Accuracy", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{model_name.upper()} — Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info("Training curves saved to %s", save_path)


# ────────────────────────────────────────────────────────────
# Main training pipeline
# ────────────────────────────────────────────────────────────
def train() -> None:
    """
    Execute the full training pipeline for both MLP and LSTM models.

    Steps:
        1. Load RAVDESS dataset.
        2. Extract features for every file.
        3. Stratified train/test split.
        4. Normalise features (StandardScaler, saved for inference).
        5. One-hot encode labels.
        6. Train MLP and LSTM, plot curves, save models.

    Returns:
        None
    """
    set_seeds()

    # 1. Load dataset
    logger.info("Loading RAVDESS dataset …")
    df = load_ravdess_dataset()
    get_class_distribution(df)

    # 2. Extract features
    X_flat, X_3d, y = extract_all_features(df)

    # 3. Train/test split (stratified)
    (
        X_flat_train, X_flat_test,
        X_3d_train, X_3d_test,
        y_train, y_test,
    ) = train_test_split(
        X_flat,
        X_3d,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Split: %d train, %d test samples.",
        len(y_train),
        len(y_test),
    )

    # 4a. Normalise flat features (MLP)
    scaler_flat = StandardScaler()
    X_flat_train = scaler_flat.fit_transform(X_flat_train)
    X_flat_test = scaler_flat.transform(X_flat_test)

    scaler_flat_path = os.path.join(config.MODEL_SAVE_PATH, "scaler_flat.pkl")
    joblib.dump(scaler_flat, scaler_flat_path)
    logger.info("Flat-feature scaler saved to %s", scaler_flat_path)

    # 4b. Normalise 3-D features (LSTM) — reshape → scale → reshape back
    n_train, t_steps, n_feat = X_3d_train.shape
    n_test = X_3d_test.shape[0]

    scaler_3d = StandardScaler()
    X_3d_train = scaler_3d.fit_transform(
        X_3d_train.reshape(-1, n_feat)
    ).reshape(n_train, t_steps, n_feat)

    X_3d_test = scaler_3d.transform(
        X_3d_test.reshape(-1, n_feat)
    ).reshape(n_test, t_steps, n_feat)

    scaler_3d_path = os.path.join(config.MODEL_SAVE_PATH, "scaler_3d.pkl")
    joblib.dump(scaler_3d, scaler_3d_path)
    logger.info("3D-feature scaler saved to %s", scaler_3d_path)

    # 5. One-hot encode labels
    y_train_cat = to_categorical(y_train, num_classes=config.NUM_CLASSES)
    y_test_cat = to_categorical(y_test, num_classes=config.NUM_CLASSES)

    # Save test data for evaluate.py
    np.savez(
        os.path.join(config.FEATURES_PATH, "test_data.npz"),
        X_flat_test=X_flat_test,
        X_3d_test=X_3d_test,
        y_test=y_test,
        y_test_cat=y_test_cat,
    )
    logger.info("Test data saved to features/test_data.npz")

    # ── 6a. Train MLP ──
    logger.info("=" * 60)
    logger.info("Training MLP model …")
    logger.info("=" * 60)

    mlp = build_mlp_model()
    compile_model(mlp)

    mlp_history = mlp.fit(
        X_flat_train,
        y_train_cat,
        validation_data=(X_flat_test, y_test_cat),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks("mlp"),
        verbose=1,
    )

    mlp_save_path = os.path.join(config.MODEL_SAVE_PATH, "mlp_model.keras")
    mlp.save(mlp_save_path)
    logger.info("MLP model saved to %s", mlp_save_path)
    plot_training_history(mlp_history, "mlp")

    # ── 6b. Train LSTM ──
    logger.info("=" * 60)
    logger.info("Training LSTM model …")
    logger.info("=" * 60)

    lstm = build_lstm_model(time_steps=t_steps, n_features=n_feat)
    compile_model(lstm)

    lstm_history = lstm.fit(
        X_3d_train,
        y_train_cat,
        validation_data=(X_3d_test, y_test_cat),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks("lstm"),
        verbose=1,
    )

    lstm_save_path = os.path.join(config.MODEL_SAVE_PATH, "lstm_model.keras")
    lstm.save(lstm_save_path)
    logger.info("LSTM model saved to %s", lstm_save_path)
    plot_training_history(lstm_history, "lstm")

    # Final evaluation summary
    mlp_loss, mlp_acc = mlp.evaluate(X_flat_test, y_test_cat, verbose=0)
    lstm_loss, lstm_acc = lstm.evaluate(X_3d_test, y_test_cat, verbose=0)

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("MLP  — Test Accuracy: %.4f | Test Loss: %.4f", mlp_acc, mlp_loss)
    logger.info("LSTM — Test Accuracy: %.4f | Test Loss: %.4f", lstm_acc, lstm_loss)
    logger.info("=" * 60)


# ────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    train()
