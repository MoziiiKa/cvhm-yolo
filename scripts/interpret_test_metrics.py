#!/usr/bin/env python3
"""
Interpret YOLOv8 test split metrics, computing F1 if missing,
and categorizing each metric against defined thresholds.
"""
import json, os, math
from pathlib import Path

# Load config
project = Path(__file__).parent.parent
cfg     = json.load(open(project / "config.json"))
DATA    = Path(os.path.expanduser(os.path.expandvars(cfg["data_root"])))
LOGS    = DATA / cfg.get("logs_dir", "logs")
json_path = LOGS / "test_metrics.json"

# Thresholds (adjust to domain needs)
THRESHOLDS = {
    "metrics/mAP50-95": {"excellent":0.70, "good":0.50, "fair":0.30},
    "metrics/mAP50":    {"excellent":0.85, "good":0.70, "fair":0.50},
    "metrics/precision":{"excellent":0.75, "good":0.60, "fair":0.40},
    "metrics/recall":   {"excellent":0.75, "good":0.60, "fair":0.40},
    "metrics/F1":       {"excellent":0.75, "good":0.60, "fair":0.40},
    "fitness":          {"excellent":0.60, "good":0.50, "fair":0.40},
}
LABELS = {
    "excellent": "Excellent ✅",
    "good":      "Good 👍",
    "fair":      "Fair 🤔",
    "poor":      "Poor ⚠️",
    None:        "Unknown"
}

def clean(name): 
    return name.split("(")[0]

def categorize(key, val):
    thr = THRESHOLDS.get(key)
    if not thr: return None
    if val >= thr["excellent"]: return "excellent"
    if val >= thr["good"]:      return "good"
    if val >= thr["fair"]:      return "fair"
    return "poor"

# Load test metrics
metrics = json.loads(json_path.read_text())

# Ensure we have precision & recall
prec = metrics.get("metrics/precision(B)")
rec  = metrics.get("metrics/recall(B)")

# Compute F1 if missing
if "metrics/F1(B)" not in metrics and prec is not None and rec is not None:
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec)>0 else 0.0
    metrics["metrics/F1(B)"] = f1

# Print report
print("\nModel Generalization Report (Test Split)\n" + "-"*44)
for raw, val in metrics.items():
    base = clean(raw)
    cat  = categorize(base, val)
    print(f"{raw:25s}: {val:.3f} → {LABELS[cat]}")
print("-"*44 + "\nInterpretation complete.\n")
