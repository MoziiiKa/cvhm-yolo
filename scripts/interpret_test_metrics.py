#!/usr/bin/env python3
"""
Interpretation of YOLOv8 test split metrics:
Loads test_metrics.json, normalizes metric names (strips any '(...)' suffix),
compares against thresholds, and prints a summary evaluation of model generalization.
"""
import json
from pathlib import Path
import os

# --- Configuration: adjust these to your standards ---
THRESHOLDS = {
    "metrics/mAP50-95": {"excellent": 0.70, "good": 0.50, "fair": 0.30},
    "metrics/mAP50":    {"excellent": 0.85, "good": 0.70, "fair": 0.50},
    "metrics/precision":{"excellent": 0.75, "good": 0.60, "fair": 0.40},
    "metrics/recall":   {"excellent": 0.75, "good": 0.60, "fair": 0.40},
    "metrics/F1":       {"excellent": 0.75, "good": 0.60, "fair": 0.40},
    "fitness":          {"excellent": 0.60, "good": 0.50, "fair": 0.40},
}


CATEGORY_NAMES = {
    "excellent": "Excellent ✅",
    "good":      "Good 👍",
    "fair":      "Fair 🤔",
    "poor":      "Poor ⚠️",
    None:        "Unknown"
}

def interpret_value(metric_key, value):
    """
    Determine performance category for a metric value.
    metric_key should be the cleaned name (without suffix).
    """
    thr = THRESHOLDS.get(metric_key)
    if not thr:
        return None
    if value >= thr["excellent"]:
        return "excellent"
    if value >= thr["good"]:
        return "good"
    if value >= thr["fair"]:
        return "fair"
    return "poor"

def clean_name(name):
    """
    Strip any parenthetical suffix, e.g. 'metrics/precision(B)' → 'metrics/precision'
    """
    return name.split("(")[0]

def main():
    project = Path(__file__).parent.parent
    # Load data_root and logs_dir from config
    cfg = json.load(open(project / "config.json"))
    data_root = Path(os.path.expanduser(os.path.expandvars(cfg["data_root"])))
    logs_dir  = data_root / cfg.get("logs_dir", "logs")
    json_path = logs_dir / "test_metrics.json"
    
    if not json_path.exists():
        print(f"ERROR: metrics file not found at {json_path}")
        return
    
    metrics = json.loads(json_path.read_text())
    print("\nModel Generalization Report (Test Split)\n" + "-"*40)
    
    for raw_name, value in metrics.items():
        base_name = clean_name(raw_name)
        category = interpret_value(base_name, value)
        label    = CATEGORY_NAMES[category]
        print(f"{raw_name:25s}: {value:.3f} → {label}")
    
    print("-"*40 + "\nInterpretation complete.\n")

if __name__ == "__main__":
    main()
