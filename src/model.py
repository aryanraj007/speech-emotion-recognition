"""
Model definitions for Speech Emotion Recognition.

Provides two architectures:
    * **Model A (MLP):** A baseline multi-layer perceptron that operates
      on the aggregated 530-dim feature vector.
    * **Model B (LSTM):** The primary recurrent model that processes raw
      MFCC frames over time.

Both models output a softmax probability distribution over
``config.NUM_CLASSES`` emotion categories.
"""

import logging

import keras
from keras import layers

from src import config

logger = logging.getLogger(__name__)


def build_mlp_model(
    input_dim: int = config.FEATURE_DIM,
    num_classes: int = config.NUM_CLASSES,
) -> keras.Model:
    """
    Build Model A — Baseline MLP for quick benchmarking.

    Architecture:
        Dense(256, relu) → BatchNorm → Dropout(0.3)
        Dense(128, relu) → BatchNorm → Dropout(0.3)
        Dense(64, relu) → Dropout(0.2)
        Dense(num_classes, softmax)

    Args:
        input_dim: Dimensionality of the input feature vector.
        num_classes: Number of emotion categories.

    Returns:
        An uncompiled ``keras.Model``.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,), name="mlp_input"),
            # Hidden layer 1
            layers.Dense(256, activation="relu", name="dense_1"),
            layers.BatchNormalization(name="bn_1"),
            layers.Dropout(0.3, name="dropout_1"),
            # Hidden layer 2
            layers.Dense(128, activation="relu", name="dense_2"),
            layers.BatchNormalization(name="bn_2"),
            layers.Dropout(0.3, name="dropout_2"),
            # Hidden layer 3
            layers.Dense(64, activation="relu", name="dense_3"),
            layers.Dropout(0.2, name="dropout_3"),
            # Output
            layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="SER_MLP",
    )

    logger.info("Built MLP model with %d parameters.", model.count_params())
    return model


def build_lstm_model(
    time_steps: int = config.LSTM_TIME_STEPS,
    n_features: int = config.N_MFCC,
    num_classes: int = config.NUM_CLASSES,
) -> keras.Model:
    """
    Build Model B — LSTM (primary model).

    Architecture:
        LSTM(128, return_sequences=True)  → Dropout(0.3)
        LSTM(64,  return_sequences=False) → Dropout(0.3)
        Dense(64, relu) → BatchNorm → Dropout(0.2)
        Dense(num_classes, softmax)

    Args:
        time_steps: Number of MFCC time frames per sample.
        n_features: Number of MFCC coefficients per frame.
        num_classes: Number of emotion categories.

    Returns:
        An uncompiled ``keras.Model``.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(time_steps, n_features), name="lstm_input"),
            # Recurrent layers
            layers.LSTM(128, return_sequences=True, name="lstm_1"),
            layers.Dropout(0.3, name="lstm_dropout_1"),
            layers.LSTM(64, return_sequences=False, name="lstm_2"),
            layers.Dropout(0.3, name="lstm_dropout_2"),
            # Dense head
            layers.Dense(64, activation="relu", name="dense_1"),
            layers.BatchNormalization(name="bn_1"),
            layers.Dropout(0.2, name="dropout_1"),
            # Output
            layers.Dense(num_classes, activation="softmax", name="output"),
        ],
        name="SER_LSTM",
    )

    logger.info("Built LSTM model with %d parameters.", model.count_params())
    return model


def compile_model(model: keras.Model, config_module=config) -> keras.Model:
    """
    Compile a Keras model with Adam optimizer and categorical cross-entropy loss.

    Args:
        model: A ``keras.Model`` (either MLP or LSTM).
        config_module: Configuration module supplying ``LEARNING_RATE``.

    Returns:
        The same ``keras.Model``, now compiled and ready for training.
    """
    optimizer = keras.optimizers.Adam(learning_rate=config_module.LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    logger.info("Compiled model '%s' with lr=%s.", model.name, config_module.LEARNING_RATE)
    return model


# ─────────────────────────────────────────────
# Quick summary when run as a script
# ─────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    mlp = build_mlp_model()
    compile_model(mlp)
    mlp.summary()

    print("\n" + "=" * 60 + "\n")

    lstm = build_lstm_model()
    compile_model(lstm)
    lstm.summary()
