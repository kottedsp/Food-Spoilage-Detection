"""
data_loader.py — Image preprocessing, augmentation, and dataset pipeline
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, AUGMENTATION, CLASSES
)


def get_generators(train_dir=TRAIN_DIR, val_dir=VAL_DIR, test_dir=TEST_DIR):
    """
    Build Keras ImageDataGenerators for train / val / test splits.
    Returns (train_gen, val_gen, test_gen).
    """
    # Training generator — with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=AUGMENTATION["rotation_range"],
        width_shift_range=AUGMENTATION["width_shift_range"],
        height_shift_range=AUGMENTATION["height_shift_range"],
        shear_range=AUGMENTATION["shear_range"],
        zoom_range=AUGMENTATION["zoom_range"],
        horizontal_flip=AUGMENTATION["horizontal_flip"],
        brightness_range=AUGMENTATION["brightness_range"],
        fill_mode="nearest",
    )

    # Validation / test generators — rescale only
    eval_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=True,
        seed=42,
    )

    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def preprocess_single_image(image_path):
    """
    Load and preprocess a single image for inference.
    Returns a (1, H, W, 3) float32 tensor in [0, 1].
    """
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def preprocess_cv2_frame(frame):
    """
    Preprocess an OpenCV BGR frame for inference.
    Returns a (1, H, W, 3) float32 numpy array.
    """
    import cv2
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    return np.expand_dims(resized.astype("float32") / 255.0, axis=0)


def get_class_weights(train_gen):
    """
    Compute class weights to handle class imbalance.
    Returns a dict {class_index: weight}.
    """
    from sklearn.utils.class_weight import compute_class_weight

    labels = train_gen.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels,
    )
    return dict(enumerate(weights))


def describe_dataset(gen, split_name="Dataset"):
    total = gen.samples
    per_class = {cls: 0 for cls in CLASSES}
    for cls, idx in gen.class_indices.items():
        count = np.sum(gen.classes == idx)
        per_class[cls] = count
    print(f"\n{split_name}: {total} images")
    for cls, count in per_class.items():
        print(f"  {cls:>10}: {count}")
