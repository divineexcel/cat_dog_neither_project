"""
Script to add the two new Komondor (mop dog) images to the dataset.
Usage: python add_mop_dogs.py --img1 /path/to/image1.jpg --img2 /path/to/image2.jpg
Or run without args to use URLs.
"""

import sys
import os
import argparse
import urllib.request
import shutil
from PIL import Image
import io

TRAIN_DOG_DIR = "data/train/dog"
VAL_DOG_DIR   = "data/val/dog"

# Komondor dog image URLs (publicly licensed / wikimedia commons)
FALLBACK_URLS = [
    # Image 1: White standing Komondor (similar to the one shared)
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Komondor_delvin.jpg/640px-Komondor_delvin.jpg",
    # Image 2: Lying Komondor (similar to the second one shared)
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Komondor_lying.jpg/640px-Komondor_lying.jpg",
]

def save_image(src, dest_path, resize=(224, 224)):
    """Open image from file path or URL, resize to 224x224, save as JPEG."""
    if src.startswith("http://") or src.startswith("https://"):
        with urllib.request.urlopen(src) as resp:
            data = resp.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    else:
        img = Image.open(src).convert("RGB")
    # Keep aspect ratio with thumbnail, then center-crop
    img.thumbnail((max(resize), max(resize)), Image.LANCZOS)
    # Center crop to exact size
    w, h = img.size
    left   = (w - resize[0]) // 2
    top    = (h - resize[1]) // 2
    right  = left + resize[0]
    bottom = top  + resize[1]
    if w < resize[0] or h < resize[1]:
        img = img.resize(resize, Image.LANCZOS)
    else:
        img = img.crop((left, top, right, bottom))
    img.save(dest_path, "JPEG", quality=92)
    print(f"  Saved → {dest_path}  ({os.path.getsize(dest_path)//1024} KB)")


def main():
    parser = argparse.ArgumentParser(description="Add mop-dog images to the dataset.")
    parser.add_argument("--img1", default=None, help="Path or URL for mop_dog_03 (standing Komondor)")
    parser.add_argument("--img2", default=None, help="Path or URL for mop_dog_04 (lying Komondor)")
    parser.add_argument("--train-only", action="store_true", help="Add to training set only")
    parser.add_argument("--val-only",   action="store_true", help="Add to validation set only")
    args = parser.parse_args()

    src1 = args.img1 if args.img1 else FALLBACK_URLS[0]
    src2 = args.img2 if args.img2 else FALLBACK_URLS[1]

    add_train = not args.val_only
    add_val   = not args.train_only

    os.makedirs(TRAIN_DOG_DIR, exist_ok=True)
    os.makedirs(VAL_DOG_DIR,   exist_ok=True)

    imgs = [
        (src1, "mop_dog_03.jpeg"),
        (src2, "mop_dog_04.jpeg"),
    ]

    for src, fname in imgs:
        print(f"\nProcessing {fname}  (source: {src[:80]}...)" if len(src) > 80 else f"\nProcessing {fname}  (source: {src})")
        if add_train:
            dest = os.path.join(TRAIN_DOG_DIR, fname)
            if os.path.exists(dest):
                print(f"  [SKIP] {dest} already exists")
            else:
                save_image(src, dest)
        if add_val:
            dest = os.path.join(VAL_DOG_DIR, fname)
            if os.path.exists(dest):
                print(f"  [SKIP] {dest} already exists")
            else:
                save_image(src, dest)

    print("\n✅ Done! New images added to dataset.")
    print(f"   Train dog count: {len(os.listdir(TRAIN_DOG_DIR))}")
    print(f"   Val   dog count: {len(os.listdir(VAL_DOG_DIR))}")


if __name__ == "__main__":
    main()
