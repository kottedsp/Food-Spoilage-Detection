"""
train.py — Full training pipeline: initial training + fine-tuning
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

from config import (
    MODEL_DIR, MODEL_PATH, EPOCHS, FINE_TUNE_EPOCHS, BATCH_SIZE, BACKBONE
)
from data_loader import get_generators, get_class_weights, describe_dataset
from model import build_model, unfreeze_for_fine_tuning


# ── Reproducibility ────────────────────────────────────────────────────────────
tf.random.set_seed(42)
np.random.seed(42)


def get_callbacks(phase="initial"):
    """Build Keras callbacks for the given training phase."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    ckpt_path = os.path.join(
        MODEL_DIR, f"best_{phase}.keras"
    )

    return [
        ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(
            log_dir=os.path.join(MODEL_DIR, "logs", phase),
            histogram_freq=1,
        ),
    ]


def plot_history(history, phase="initial", save_dir=MODEL_DIR):
    """Save accuracy and loss curves."""
    os.makedirs(save_dir, exist_ok=True)
    h = history.history
    epochs = range(1, len(h["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training curves — {phase} phase", fontsize=14)

    # Accuracy
    axes[0].plot(epochs, h["accuracy"],     label="Train acc")
    axes[0].plot(epochs, h["val_accuracy"], label="Val acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(epochs, h["loss"],     label="Train loss")
    axes[1].plot(epochs, h["val_loss"], label="Val loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True)

    path = os.path.join(save_dir, f"curves_{phase}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves → {path}")


def save_history(history, phase="initial"):
    path = os.path.join(MODEL_DIR, f"history_{phase}.json")
    with open(path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    print(f"Saved history → {path}")


def train():
    print("=" * 60)
    print(f"  Food Spoilage Detection — Training ({BACKBONE})")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_gen, val_gen, test_gen = get_generators()
    describe_dataset(train_gen, "Train")
    describe_dataset(val_gen,   "Val")
    describe_dataset(test_gen,  "Test")

    class_weights = get_class_weights(train_gen)
    print(f"\nClass weights: {class_weights}")

    # ── Phase 1: Feature extraction (backbone frozen) ─────────────────────────
    print("\n── Phase 1: Feature extraction ──")
    model, base_model = build_model()
    model.summary(line_length=90)

    history1 = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks("initial"),
        class_weight=class_weights,
        verbose=1,
    )
    plot_history(history1, "initial")
    save_history(history1, "initial")

    # ── Phase 2: Fine-tuning ──────────────────────────────────────────────────
    print("\n── Phase 2: Fine-tuning ──")
    model = unfreeze_for_fine_tuning(model, base_model)

    history2 = model.fit(
        train_gen,
        epochs=FINE_TUNE_EPOCHS,
        validation_data=val_gen,
        callbacks=get_callbacks("finetune"),
        class_weight=class_weights,
        verbose=1,
    )
    plot_history(history2, "finetune")
    save_history(history2, "finetune")

    # ── Save final model ──────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"\n✓ Final model saved → {MODEL_PATH}")

    # ── Evaluate on test set ──────────────────────────────────────────────────
    print("\n── Test-set evaluation ──")
    results = model.evaluate(test_gen, verbose=1)
    metrics = dict(zip(model.metrics_names, results))
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k:>12}: {v:.4f}")

    return model


if __name__ == "__main__":
    train()
