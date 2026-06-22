"""
app.py — Flask web interface for Food Spoilage Detection
Run: python app.py
"""

import os
import sys
import io
import base64
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf

from src.config import (
    MODEL_PATH, CLASSES, IMG_SIZE,
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    UPLOAD_FOLDER, ALLOWED_EXTENSIONS, MAX_CONTENT_LENGTH
)

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model once at startup ─────────────────────────────────────────────────
model = None


def get_model():
    global model
    if model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. "
                "Please run: python src/train.py"
            )
        print("[INFO] Loading model …")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("[INFO] Model loaded.")
    return model


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """Resize and normalise a PIL image to (1, H, W, 3) float32."""
    img   = img.convert("RGB").resize(IMG_SIZE)
    arr   = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)


def pil_to_b64(img: Image.Image, max_dim=400) -> str:
    """Return a base64-encoded JPEG thumbnail for embedding in JSON."""
    img.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"File type not allowed. Use: {ALLOWED_EXTENSIONS}"}), 400

    try:
        img   = Image.open(file.stream)
        inp   = preprocess_pil(img)
        mdl   = get_model()

        t0    = time.perf_counter()
        probs = mdl.predict(inp, verbose=0)[0]
        dt    = (time.perf_counter() - t0) * 1000   # ms

        idx   = int(np.argmax(probs))
        label = CLASSES[idx]
        conf  = float(probs[idx])

        class_probs = {cls: float(probs[i]) for i, cls in enumerate(CLASSES)}
        thumbnail   = pil_to_b64(img)

        return jsonify({
            "prediction"   : label,
            "confidence"   : round(conf * 100, 2),
            "class_probs"  : {k: round(v * 100, 2) for k, v in class_probs.items()},
            "inference_ms" : round(dt, 1),
            "thumbnail"    : thumbnail,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    model_loaded = model is not None
    return jsonify({
        "status"      : "ok",
        "model_loaded": model_loaded,
        "classes"     : CLASSES,
    })


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        get_model()
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("[WARNING] Starting server anyway — train the model before uploading images.\n")

    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
    )
