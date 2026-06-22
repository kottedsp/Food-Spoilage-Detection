# Food-Spoilage-Detection


A complete end-to-end CNN pipeline that classifies food images as **Fresh** or **Spoiled** using transfer learning (MobileNetV2), with a real-time OpenCV predictor and a Flask web interface.

---

## Project Structure

```
food_spoilage_detection/
├── app.py                    # Flask web application
├── download_dataset.py       # Dataset downloader / synthetic data generator
├── requirements.txt
│
├── src/
│   ├── config.py             # All hyperparameters and paths (edit here)
│   ├── data_loader.py        # Preprocessing, augmentation, data pipeline
│   ├── model.py              # CNN model builder (transfer learning + custom CNN)
│   ├── train.py              # Training script (Phase 1 + Fine-tuning)
│   ├── evaluate.py           # Confusion matrix, ROC, misclassified samples
│   └── predict.py            # Real-time webcam / image / batch inference
│
├── data/
│   ├── train/{fresh,spoiled}/
│   ├── val/{fresh,spoiled}/
│   └── test/{fresh,spoiled}/
│
├── models/                   # Saved models, curves, logs (created at runtime)
├── notebooks/
│   └── exploration.ipynb     # Interactive walkthrough
├── templates/
│   └── index.html            # Web UI
└── static/
    └── uploads/              # Temporary upload storage
```

---

## Quick Start

### 1. Create a virtual environment

```bash
cd food_spoilage_detection
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get the dataset

**Option A — Kaggle (recommended, real food images):**
```bash
# Install kaggle CLI and place kaggle.json in ~/.kaggle/
pip install kaggle
python download_dataset.py
```

**Option B — Synthetic data (no Kaggle account needed, for testing):**
```bash
python download_dataset.py --synthetic --n 300
```

**Option C — Bring your own data:**
Place images into `data/train/fresh/`, `data/train/spoiled/`, and equivalents for `val/` and `test/`.

### 3. Train the model

```bash
python src/train.py
```

This runs two phases:
- **Phase 1** – feature extraction with frozen MobileNetV2 backbone
- **Phase 2** – fine-tuning of the top layers

Training curves, model checkpoints, and logs are saved to `models/`.

### 4. Evaluate

```bash
python src/evaluate.py
```

Outputs: classification report, confusion matrix, ROC curve, misclassified sample grid — all saved as PNGs in `models/`.

### 5. Real-time prediction

```bash
# Webcam live feed
python src/predict.py

# Single image
python src/predict.py --image path/to/apple.jpg

# Batch folder
python src/predict.py --folder path/to/images/
```

### 6. Web interface

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser, upload a food photo, and get an instant freshness assessment.

---

## Configuration

All settings live in **`src/config.py`**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BACKBONE` | `MobileNetV2` | Pre-trained backbone (`EfficientNetB0`, `ResNet50`) |
| `EPOCHS` | `30` | Max epochs for Phase 1 |
| `FINE_TUNE_EPOCHS` | `10` | Fine-tuning epochs |
| `LEARNING_RATE` | `1e-4` | Initial learning rate |
| `BATCH_SIZE` | `32` | Batch size |
| `DROPOUT_RATE` | `0.4` | Dropout for the classification head |
| `IMG_SIZE` | `(224, 224)` | Input resolution |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data Augmentation | Keras ImageDataGenerator |
| Backbone | MobileNetV2 (ImageNet pre-trained) |
| Metrics | scikit-learn |
| Web API | Flask |
| Visualisation | Matplotlib, Seaborn |

---

## Model Architecture

```
Input (224×224×3)
   └── MobileNetV2 (frozen → then fine-tuned)
         └── GlobalAveragePooling2D
               └── Dense(256, ReLU) + BatchNorm + Dropout
                     └── Dense(128, ReLU)
                           └── Dense(2, Softmax)
```

---

## Results (example with real dataset)

| Metric | Value |
|--------|-------|
| Train accuracy | ~97% |
| Val accuracy | ~94% |
| Test accuracy | ~93% |
| ROC-AUC | ~0.97 |

*Results vary by dataset and hardware.*
