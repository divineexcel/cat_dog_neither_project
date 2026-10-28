"""Validation metrics shared by training and the Streamlit app."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def model_sha256(path: str | Path) -> str:
    """Return a stable fingerprint for the model used by a metrics report."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    """Calculate the confusion matrix and class metrics from integer labels."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n_classes = len(class_names)

    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)

    per_class = []
    for index, name in enumerate(class_names):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        support = int(matrix[index, :].sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({
            "name": name,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1": round(f1 * 100, 1),
            "support": support,
        })

    return {
        "class_names": list(class_names),
        "confusion_matrix": matrix.tolist(),
        "accuracy": round(float(np.trace(matrix) / matrix.sum()) * 100, 1) if matrix.sum() else 0.0,
        "precision": round(float(np.mean([item["precision"] for item in per_class])), 1),
        "recall": round(float(np.mean([item["recall"] for item in per_class])), 1),
        "f1": round(float(np.mean([item["f1"] for item in per_class])), 1),
        "per_class": per_class,
        "n_val": int(matrix.sum()),
    }


def evaluate_model(model, val_ds, class_names: list[str]) -> dict:
    """Evaluate a Keras model on a labelled validation dataset."""
    y_true_parts, y_pred_parts = [], []
    for batch_x, batch_y in val_ds:
        probabilities = model.predict(batch_x, verbose=0)
        y_pred_parts.append(np.argmax(probabilities, axis=1))
        labels = batch_y.numpy()
        y_true_parts.append(np.argmax(labels, axis=1) if labels.ndim > 1 else labels)

    if not y_true_parts:
        raise ValueError("The validation dataset contains no images.")
    return build_metrics(np.concatenate(y_true_parts), np.concatenate(y_pred_parts), class_names)


def save_metrics_report(report: dict, model_path: str | Path, report_path: str | Path) -> None:
    """Save metrics beside the model with a fingerprint to prevent stale results."""
    report = {**report, "model_sha256": model_sha256(model_path)}
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
