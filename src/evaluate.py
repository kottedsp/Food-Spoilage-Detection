"""
evaluate.py — Detailed model evaluation: confusion matrix, classification report,
               per-class accuracy, and misclassified sample display.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from config import MODEL_PATH, MODEL_DIR, CLASSES, IMG_SIZE
from data_loader import get_generators


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train.py first."
        )
    print(f"Loading model from {path} …")
    return tf.keras.models.load_model(path)


def plot_confusion_matrix(cm, class_names, save_dir=MODEL_DIR):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {path}")


def plot_roc(y_true, y_prob, save_dir=MODEL_DIR):
    """ROC curve for binary spoilage detection."""
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Spoilage Detection")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved ROC curve → {path}")
    return roc_auc


def show_misclassified(model, test_gen, n=9, save_dir=MODEL_DIR):
    """Display up to n misclassified images."""
    all_images, all_true, all_pred = [], [], []

    for images, labels in test_gen:
        preds = model.predict(images, verbose=0)
        all_images.append(images)
        all_true.extend(np.argmax(labels, axis=1))
        all_pred.extend(np.argmax(preds,  axis=1))
        if len(all_true) >= test_gen.samples:
            break

    all_images = np.concatenate(all_images, axis=0)[:test_gen.samples]
    all_true   = np.array(all_true[:test_gen.samples])
    all_pred   = np.array(all_pred[:test_gen.samples])

    wrong = np.where(all_true != all_pred)[0]
    if len(wrong) == 0:
        print("No misclassified samples on the test set!")
        return

    sample = wrong[:n]
    cols   = min(3, len(sample))
    rows   = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, idx in enumerate(sample):
        img = all_images[idx]
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {CLASSES[all_true[idx]]}\nPred: {CLASSES[all_pred[idx]]}",
            color="red", fontsize=9
        )
        axes[i].axis("off")

    for j in range(len(sample), len(axes)):
        axes[j].axis("off")

    plt.suptitle("Misclassified Samples", fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "misclassified.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved misclassified grid → {path}")


def evaluate():
    _, _, test_gen = get_generators()
    model = load_model()

    print("\nRunning predictions on test set …")
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes

    # Classification report
    print("\n── Classification Report ──")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, CLASSES)

    # ROC + AUC
    roc_auc = plot_roc(y_true, y_prob)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Misclassified samples
    show_misclassified(model, test_gen)

    print("\n✓ Evaluation complete. Check models/ for saved plots.")


if __name__ == "__main__":
    evaluate()
