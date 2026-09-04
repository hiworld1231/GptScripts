#!/usr/bin/env python3
import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

DATASET = Path("skillcheck_dataset.jsonl")
MODEL = Path("skillcheck_model.json")
FEATURES = [
    "white_strength", "black_strength", "white_peak", "black_peak",
    "red_strength", "red_peak", "edge_strength", "edge_peak",
    "white_distance_deg", "black_distance_deg", "target_distance_deg",
    "speed_deg_s", "fit_residual_deg", "radius_mean", "frames",
]


def atomic_write(path, data):
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def feature_vector(summary):
    red = summary.get("red_prediction") or {}
    values = []
    for key in FEATURES:
        if key in red:
            value = red.get(key)
        else:
            value = summary.get(key)
        if value is None or not math.isfinite(float(value)):
            values.append(0.0)
        else:
            values.append(float(value))
    return values


def collect_labeled():
    path = Path("analysis_all.json")
    if not path.exists():
        return []
    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out = []
    for session in root.get("sessions", []):
        for check in session.get("checks", []):
            summary = check.get("summary", {})
            label = summary.get("ground_truth")
            if label not in {"WHITE", "BLACK"}:
                continue
            out.append({
                "id": f"{session.get('file','')}:{summary.get('index')}",
                "label": label,
                "features": feature_vector(summary),
            })
    return out


def sigmoid(x):
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def train(samples):
    if len(samples) < 4:
        return None
    labels = np.array([1.0 if x["label"] == "WHITE" else 0.0 for x in samples], dtype=np.float64)
    if len(np.unique(labels)) < 2:
        return None
    X = np.array([x["features"] for x in samples], dtype=np.float64)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-9] = 1.0
    Z = (X - mean) / scale
    Z = np.column_stack([np.ones(len(Z)), Z])
    w = np.zeros(Z.shape[1], dtype=np.float64)
    for _ in range(1800):
        p = sigmoid(Z @ w)
        grad = (Z.T @ (p - labels)) / len(Z)
        grad[1:] += 0.01 * w[1:]
        w -= 0.08 * grad
    pred = sigmoid(Z @ w) >= 0.5
    acc = float(np.mean(pred == labels))
    return {
        "version": 1,
        "features": FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": w.tolist(),
        "samples": len(samples),
        "white": int(labels.sum()),
        "black": int(len(labels) - labels.sum()),
        "training_accuracy": acc,
    }


def update_dataset(samples):
    existing = {}
    if DATASET.exists():
        for line in DATASET.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                existing[row["id"]] = row
            except Exception:
                pass
    for row in samples:
        existing[row["id"]] = row
    with open(DATASET, "w", encoding="utf-8") as f:
        for key in sorted(existing):
            f.write(json.dumps(existing[key], ensure_ascii=False, separators=(",", ":")) + "\n")
    return list(existing.values())


def run():
    samples = collect_labeled()
    if not samples:
        print("AI: labeled samples = 0")
        return 0
    samples = update_dataset(samples)
    model = train(samples)
    if model is None:
        print(f"AI: samples={len(samples)}, model пока не обучен: нужны WHITE и BLACK")
        return 0
    atomic_write(MODEL, model)
    print(f"AI: trained samples={model['samples']} WHITE={model['white']} BLACK={model['black']} train_accuracy={model['training_accuracy']:.3f}")
    return 0


def predict(summary):
    if not MODEL.exists():
        return None
    try:
        model = json.loads(MODEL.read_text(encoding="utf-8"))
        x = np.array(feature_vector(summary), dtype=np.float64)
        mean = np.array(model["mean"], dtype=np.float64)
        scale = np.array(model["scale"], dtype=np.float64)
        w = np.array(model["weights"], dtype=np.float64)
        z = (x - mean) / scale
        p = float(sigmoid(w[0] + z @ w[1:]))
        label = "WHITE" if p >= 0.5 else "BLACK"
        confidence = p if label == "WHITE" else 1.0 - p
        return {"prediction": label, "confidence": confidence, "white_probability": p, "samples": model.get("samples", 0)}
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    if args.train:
        raise SystemExit(run())
    raise SystemExit(run())


if __name__ == "__main__":
    main()
