"""
config.py — Central configuration for Food Spoilage Detection
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")
VAL_DIR    = os.path.join(DATA_DIR, "val")
TEST_DIR   = os.path.join(DATA_DIR, "test")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "food_spoilage_model.keras")

# ── Classes ────────────────────────────────────────────────────────────────────
CLASSES      = ["fresh", "spoiled"]
NUM_CLASSES  = len(CLASSES)

# ── Image settings ─────────────────────────────────────────────────────────────
IMG_HEIGHT = 224
IMG_WIDTH  = 224
IMG_SIZE   = (IMG_HEIGHT, IMG_WIDTH)

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 1e-4
DROPOUT_RATE  = 0.4

# ── Transfer-learning backbone ─────────────────────────────────────────────────
BACKBONE        = "MobileNetV2"   # Options: "MobileNetV2", "EfficientNetB0", "ResNet50"
FINE_TUNE_AT    = 100             # Unfreeze layers from this index onward
FINE_TUNE_EPOCHS = 10

# ── Augmentation ───────────────────────────────────────────────────────────────
AUGMENTATION = {
    "rotation_range"    : 20,
    "width_shift_range" : 0.15,
    "height_shift_range": 0.15,
    "shear_range"       : 0.1,
    "zoom_range"        : 0.2,
    "horizontal_flip"   : True,
    "brightness_range"  : [0.8, 1.2],
}

# ── Flask web app ───────────────────────────────────────────────────────────────
FLASK_HOST        = "0.0.0.0"
FLASK_PORT        = 5000
FLASK_DEBUG       = False
UPLOAD_FOLDER     = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024   # 16 MB
