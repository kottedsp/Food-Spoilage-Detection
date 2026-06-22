"""
predict.py — Real-time food spoilage detection via webcam or image file.

Usage:
    # Webcam (press 'q' to quit, 's' to save snapshot)
    python src/predict.py

    # Single image
    python src/predict.py --image path/to/food.jpg

    # Batch folder
    python src/predict.py --folder path/to/images/
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from config import MODEL_PATH, CLASSES, IMG_SIZE, MODEL_DIR
from data_loader import preprocess_cv2_frame, preprocess_single_image


# ── Colour palette (BGR) ───────────────────────────────────────────────────────
COLORS = {
    "fresh"  : (34,  139, 34),    # forest green
    "spoiled": (0,   0,   220),   # red
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Model not found at {path}. Run train.py first.")
    print(f"[INFO] Loading model …")
    return tf.keras.models.load_model(path)


def predict_frame(model, frame):
    """Run inference on one OpenCV frame. Returns (label, confidence, probs)."""
    inp   = preprocess_cv2_frame(frame)
    probs = model.predict(inp, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx]), probs


def annotate_frame(frame, label, confidence, probs, fps=None):
    """Draw prediction overlay on an OpenCV frame (in-place)."""
    h, w = frame.shape[:2]
    color = COLORS[label]

    # Border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)

    # Background strip
    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)

    # Prediction text
    label_text = f"{label.upper()}  {confidence * 100:.1f}%"
    cv2.putText(frame, label_text, (10, 45), FONT, 1.3, color, 2, cv2.LINE_AA)

    # Per-class confidence bars
    bar_x, bar_y, bar_h, bar_max_w = 10, 80, 18, 200
    for i, cls in enumerate(CLASSES):
        pct  = float(probs[i])
        bw   = int(pct * bar_max_w)
        bcol = COLORS[cls]
        y    = bar_y + i * (bar_h + 8)
        cv2.rectangle(frame, (bar_x, y), (bar_x + bar_max_w, y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (bar_x, y), (bar_x + bw,        y + bar_h), bcol,         -1)
        cv2.putText(frame, f"{cls} {pct * 100:.1f}%",
                    (bar_x + bar_max_w + 8, y + bar_h - 3),
                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # FPS
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, h - 10),
                    FONT, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# ── Modes ──────────────────────────────────────────────────────────────────────

def run_webcam(model, cam_index=0):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open camera {cam_index}.")

    print("[INFO] Webcam active. Press 'q' to quit, 's' to save snapshot.")
    snapshot_dir = os.path.join(MODEL_DIR, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)

    prev_t = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label, conf, probs = predict_frame(model, frame)

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-9)
        prev_t = now

        annotate_frame(frame, label, conf, probs, fps=fps)
        cv2.imshow("Food Spoilage Detector — press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            ts   = time.strftime("%Y%m%d_%H%M%S")
            path = os.path.join(snapshot_dir, f"snapshot_{ts}.jpg")
            cv2.imwrite(path, frame)
            print(f"[INFO] Snapshot saved → {path}")

    cap.release()
    cv2.destroyAllWindows()


def run_image(model, image_path):
    if not os.path.exists(image_path):
        sys.exit(f"[ERROR] Image not found: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        sys.exit(f"[ERROR] Cannot read image: {image_path}")

    label, conf, probs = predict_frame(model, frame)

    print(f"\n{'─' * 40}")
    print(f"  Image  : {os.path.basename(image_path)}")
    print(f"  Result : {label.upper()}  ({conf * 100:.1f}%)")
    for i, cls in enumerate(CLASSES):
        print(f"    {cls:>10}: {probs[i] * 100:.2f}%")
    print(f"{'─' * 40}\n")

    display = cv2.resize(frame, (640, 480))
    annotate_frame(display, label, conf, probs)
    cv2.imshow(f"Result: {label} — press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_batch(model, folder):
    exts   = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    images = [f for f in os.listdir(folder)
              if os.path.splitext(f)[1].lower() in exts]
    if not images:
        sys.exit(f"[ERROR] No images found in {folder}")

    print(f"\nBatch prediction — {len(images)} images\n{'─' * 50}")
    results = []
    for fname in sorted(images):
        path  = os.path.join(folder, fname)
        inp   = preprocess_single_image(path)
        probs = model.predict(inp, verbose=0)[0]
        idx   = int(np.argmax(probs))
        lbl   = CLASSES[idx]
        conf  = float(probs[idx])
        results.append((fname, lbl, conf))
        print(f"  {fname:<35}  {lbl:<8}  {conf * 100:.1f}%")

    fresh_ct   = sum(1 for _, l, _ in results if l == "fresh")
    spoiled_ct = len(results) - fresh_ct
    print(f"\nSummary: {fresh_ct} fresh, {spoiled_ct} spoiled out of {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Food Spoilage Detector")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--image",  type=str, help="Path to a single image")
    group.add_argument("--folder", type=str, help="Path to a folder of images")
    group.add_argument("--cam",    type=int, default=0, help="Camera index (default 0)")
    args = parser.parse_args()

    model = load_model()

    if args.image:
        run_image(model, args.image)
    elif args.folder:
        run_batch(model, args.folder)
    else:
        run_webcam(model, cam_index=args.cam)


if __name__ == "__main__":
    main()
