#!/usr/bin/env python3
"""
Interpretation of YOLOv8 test split metrics:
Loads test_metrics.json, compares against thresholds,
and prints a summary evaluation of model generalization.
"""
import json
from pathlib import Path

# --- Configuration: adjust these to your standards ---
THRESHOLDS = {
    "metrics/mAP50-95": {
        "excellent": 0.70,
        "good":      0.50,
        "fair":      0.30
    },
    "metrics/precision": {
        "excellent": 0.75,
        "good":      0.60,
        "fair":      0.40
    },
    "metrics/recall": {
        "excellent": 0.75,
        "good":      0.60,
        "fair":      0.40
    },
    "metrics/F1": {
        "excellent": 0.75,
        "good":      0.60,
        "fair":      0.40
    },
}

CATEGORY_NAMES = {
    "excellent": "Excellent ✅",
    "good":      "Good 👍",
    "fair":      "Fair 🤔",
    "poor":      "Poor ⚠️"
}

def interpret_value(metric_name, value):
    """Return a category based on THRESHOLDS for this metric."""
    thr = THRESHOLDS.get(metric_name, {})
    if not thr:
        return None
    if value >= thr["excellent"]:
        return "excellent"
    if value >= thr["good"]:
        return "good"
    if value >= thr["fair"]:
        return "fair"
    return "poor"

def main():
    root = Path(__file__).parent.parent
    data_root = root  # adjust if you store JSON under data_root/logs
    json_path = data_root / "logs" / "test_metrics.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found.")
        return

    metrics = json.loads(json_path.read_text())
    print("\nModel Generalization Report (Test Split)\n" + "-"*40)
    for name, value in metrics.items():
        category = interpret_value(name, value)
        label = CATEGORY_NAMES.get(category, "Unknown")
        print(f"{name:20s}: {value:.3f} → {label}")
    print("-"*40 + "\nInterpretation complete.\n")

if __name__ == "__main__":
    main()
