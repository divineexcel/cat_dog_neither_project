import os
import sys
import numpy as np
from PIL import Image, ImageOps
import keras
from tabulate import tabulate
from image_preprocessing import model_views

MODEL_PATH = "models/classifier.keras"
TARGET_SIZE = (224, 224)
CLASSES = ["cat", "dog", "neither"]  # Matches training alphabetical order

def classify_folder(folder_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Could not find model at {MODEL_PATH}")
        return

    # Load model once
    model = keras.models.load_model(MODEL_PATH)
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted([f for f in os.listdir(folder_path) if os.path.splitext(f.lower())[1] in valid_exts])

    if not files:
        print(f"No valid images found in: {folder_path}")
        return

    table_data = []

    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            # Handle any resolution, orientation, or channels safely
            with Image.open(filepath) as img:
                img = ImageOps.exif_transpose(img).convert("RGB")
                views = model_views(img, TARGET_SIZE)
                img_array = np.asarray(views, dtype=np.float32)
            # The saved model includes MobileNetV2 preprocessing internally.
            # Average aspect-preserving views for wide or tall scene images.
            probs = model.predict(img_array, verbose=0).mean(axis=0)
            top_idx = int(np.argmax(probs))
            pred_class = CLASSES[top_idx]
            confidence = probs[top_idx] * 100.0

            table_data.append([filename, pred_class, f"{confidence:.1f}%"])
        except Exception as e:
            table_data.append([filename, "error", str(e)])

    print("\n" + tabulate(table_data, headers=["filename", "prediction", "confidence"], tablefmt="github") + "\n")

if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "test_folder"
    classify_folder(folder)
