"""
model.py — CNN model builder using transfer learning (MobileNetV2 default)
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from config import (
    IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES,
    LEARNING_RATE, DROPOUT_RATE, BACKBONE, FINE_TUNE_AT
)


def _get_backbone(name, input_shape):
    """Return a pre-trained backbone with ImageNet weights, top excluded."""
    kwargs = dict(include_top=False, weights="imagenet", input_shape=input_shape)
    backbones = {
        "MobileNetV2"    : tf.keras.applications.MobileNetV2,
        "EfficientNetB0" : tf.keras.applications.EfficientNetB0,
        "ResNet50"       : tf.keras.applications.ResNet50,
    }
    if name not in backbones:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(backbones)}")
    base = backbones[name](**kwargs)
    base.trainable = False          # freeze initially
    return base


def build_model(backbone_name=BACKBONE, num_classes=NUM_CLASSES):
    """
    Build a transfer-learning CNN:
      Input → Backbone (frozen) → GlobalAvgPool → Dense → Dropout → Softmax
    """
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    inputs      = layers.Input(shape=input_shape, name="input")

    # ── Backbone ──────────────────────────────────────────────────────────────
    base = _get_backbone(backbone_name, input_shape)
    x    = base(inputs, training=False)

    # ── Classification head ───────────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense_256")(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4),
                     name="dense_128")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name=f"FoodSpoilage_{backbone_name}")
    _compile(model)
    return model, base


def _compile(model, lr=LEARNING_RATE):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy",
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )


def unfreeze_for_fine_tuning(model, base_model, fine_tune_at=FINE_TUNE_AT,
                              lr=LEARNING_RATE / 10):
    """Unfreeze layers from fine_tune_at onward and recompile with a lower LR."""
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    _compile(model, lr=lr)
    print(f"Fine-tuning from layer {fine_tune_at} / {len(base_model.layers)}")
    trainable_count = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )
    print(f"Trainable params: {trainable_count:,}")
    return model


def build_custom_cnn(num_classes=NUM_CLASSES):
    """
    Lightweight custom CNN (no pre-trained weights) — for experimentation.
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(num_classes, activation="softmax"),
    ], name="FoodSpoilage_CustomCNN")

    _compile(model)
    return model
