"""
download_dataset.py — Download and organise a food freshness dataset.

Option A (default): Downloads the "Fruits Fresh and Rotten" dataset from Kaggle.
Option B          : Creates a tiny synthetic dataset using random noise images
                    so the rest of the pipeline can be tested without real data.

Usage:
    python download_dataset.py              # tries Kaggle; falls back to synthetic
    python download_dataset.py --synthetic  # always use synthetic data
    python download_dataset.py --kaggle-key <username>:<key>
"""

import os
import sys
import shutil
import random
import argparse
import numpy as np
from PIL import Image

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
SPLITS    = ["train", "val", "test"]
CLASSES   = ["fresh", "spoiled"]
SPLIT_PCT = {"train": 0.70, "val": 0.15, "test": 0.15}


# ── Synthetic data generator ───────────────────────────────────────────────────

def _colour_for(cls):
    """Return a rough mean colour that visually hints at freshness."""
    if cls == "fresh":
        return (random.randint(60, 130), random.randint(140, 220), random.randint(40, 100))
    else:
        return (random.randint(120, 180), random.randint(90, 140), random.randint(30, 80))


def generate_synthetic(n_per_class=200, img_size=(224, 224)):
    """
    Create synthetic RGB images with noise + colour hint.
    This is for pipeline testing only — accuracy will be low.
    """
    print("[INFO] Generating synthetic dataset …")
    for split in SPLITS:
        for cls in CLASSES:
            d = os.path.join(DATA_DIR, split, cls)
            os.makedirs(d, exist_ok=True)

    all_paths = {cls: [] for cls in CLASSES}

    for cls in CLASSES:
        tmp = os.path.join(DATA_DIR, "_tmp", cls)
        os.makedirs(tmp, exist_ok=True)
        mean_col = _colour_for(cls)

        for i in range(n_per_class):
            arr = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
            # Add colour bias
            for c, mc in enumerate(mean_col):
                arr[:, :, c] = np.clip(arr[:, :, c] // 2 + mc // 2, 0, 255)
            # Random "blemish" circles for spoiled
            if cls == "spoiled":
                for _ in range(random.randint(3, 8)):
                    cx, cy = random.randint(0, img_size[1]), random.randint(0, img_size[0])
                    r = random.randint(10, 30)
                    y_, x_ = np.ogrid[:img_size[0], :img_size[1]]
                    mask = (x_ - cx) ** 2 + (y_ - cy) ** 2 <= r ** 2
                    arr[mask] = [random.randint(30, 70)] * 3

            path = os.path.join(tmp, f"{cls}_{i:04d}.jpg")
            Image.fromarray(arr).save(path, quality=90)
            all_paths[cls].append(path)

    # Split
    for cls in CLASSES:
        paths = all_paths[cls]
        random.shuffle(paths)
        n     = len(paths)
        cuts  = {
            "train": int(n * SPLIT_PCT["train"]),
            "val"  : int(n * SPLIT_PCT["val"]),
        }
        splits_map = {
            "train": paths[:cuts["train"]],
            "val"  : paths[cuts["train"]:cuts["train"] + cuts["val"]],
            "test" : paths[cuts["train"] + cuts["val"]:],
        }
        for split, ps in splits_map.items():
            dst = os.path.join(DATA_DIR, split, cls)
            for p in ps:
                shutil.copy(p, os.path.join(dst, os.path.basename(p)))
            print(f"  {split}/{cls}: {len(ps)} images")

    shutil.rmtree(os.path.join(DATA_DIR, "_tmp"), ignore_errors=True)
    print("[INFO] Synthetic dataset ready.\n")


# ── Kaggle download ────────────────────────────────────────────────────────────

def download_kaggle(dataset="sriramr/fruits-fresh-and-rotten-for-classification"):
    """
    Download via the Kaggle API. Requires ~/.kaggle/kaggle.json or
    KAGGLE_USERNAME / KAGGLE_KEY environment variables.
    """
    try:
        import kaggle  # noqa
    except ImportError:
        print("[INFO] kaggle package not installed. Installing …")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        import kaggle  # noqa

    print(f"[INFO] Downloading Kaggle dataset: {dataset}")
    raw_dir = os.path.join(DATA_DIR, "_kaggle_raw")
    os.makedirs(raw_dir, exist_ok=True)

    import kaggle.api
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset, path=raw_dir, unzip=True)

    # Reorganise into data/{split}/{class}/
    _reorganise_kaggle(raw_dir)
    shutil.rmtree(raw_dir, ignore_errors=True)
    print("[INFO] Kaggle dataset ready.\n")


def _reorganise_kaggle(raw_dir):
    """
    The Kaggle 'fruits-fresh-and-rotten' dataset has:
      dataset/
        Train/
          freshapples/, rottenapples/, freshbanana/, rottenbanana/, …
        Test/
          freshapples/, rottenapples/, …

    We map fresh* → fresh, rotten* → spoiled and split Test into val + test.
    """
    import glob

    def label(folder_name):
        name = folder_name.lower()
        if name.startswith("fresh"):    return "fresh"
        if name.startswith("rotten"):   return "spoiled"
        return None

    for split in SPLITS:
        for cls in CLASSES:
            os.makedirs(os.path.join(DATA_DIR, split, cls), exist_ok=True)

    # Train
    for cat_dir in glob.glob(os.path.join(raw_dir, "**", "Train", "*"), recursive=False):
        cls = label(os.path.basename(cat_dir))
        if cls is None: continue
        for img in glob.glob(os.path.join(cat_dir, "*")):
            shutil.copy(img, os.path.join(DATA_DIR, "train", cls, os.path.basename(img)))

    # Test → split into val + test
    test_imgs = {cls: [] for cls in CLASSES}
    for cat_dir in glob.glob(os.path.join(raw_dir, "**", "Test", "*"), recursive=False):
        cls = label(os.path.basename(cat_dir))
        if cls is None: continue
        test_imgs[cls].extend(glob.glob(os.path.join(cat_dir, "*")))

    for cls, imgs in test_imgs.items():
        random.shuffle(imgs)
        half = len(imgs) // 2
        for img in imgs[:half]:
            shutil.copy(img, os.path.join(DATA_DIR, "val",  cls, os.path.basename(img)))
        for img in imgs[half:]:
            shutil.copy(img, os.path.join(DATA_DIR, "test", cls, os.path.basename(img)))

    # Summary
    for split in SPLITS:
        for cls in CLASSES:
            n = len(os.listdir(os.path.join(DATA_DIR, split, cls)))
            print(f"  {split}/{cls}: {n} images")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic",  action="store_true",
                        help="Always generate synthetic data (no Kaggle needed)")
    parser.add_argument("--n",          type=int, default=200,
                        help="Images per class for synthetic mode (default 200)")
    parser.add_argument("--kaggle-key", type=str, default=None,
                        help="Kaggle credentials as username:key")
    args = parser.parse_args()

    if args.kaggle_key:
        user, key = args.kaggle_key.split(":", 1)
        os.environ["KAGGLE_USERNAME"] = user
        os.environ["KAGGLE_KEY"]      = key

    if args.synthetic:
        generate_synthetic(n_per_class=args.n)
        return

    # Try Kaggle, fall back to synthetic
    try:
        download_kaggle()
    except Exception as e:
        print(f"[WARNING] Kaggle download failed: {e}")
        print("[INFO] Falling back to synthetic data …\n")
        generate_synthetic(n_per_class=args.n)


if __name__ == "__main__":
    main()
