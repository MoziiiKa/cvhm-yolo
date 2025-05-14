#!/usr/bin/env python3
"""
Runs a full validation pass using YOLOv8 and extracts quantitative metrics
(mAP@[.5:.95], precision, recall, F1) directly from the DetMetrics object,
saving both logs and metrics JSON under the configured data root.
"""
import json
import logging
import os
from pathlib import Path
from ultralytics import YOLO

# Load config & resolve data_root
ROOT      = Path(__file__).parent.parent
cfg       = json.load(open(ROOT / "config.json"))
raw_root  = cfg["data_root"]
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))

# Setup logging directory
LOG_DIR = DATA_ROOT / cfg.get("logs_dir", "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "evaluate_metrics.log"),
        logging.StreamHandler()
    ]
)

# Paths to model weights and dataset config
WEIGHTS   = DATA_ROOT / "runs" / "exp" / "weights" / "best.pt"
DATA_YAML = ROOT / "data.yaml"

# Validate existence
if not WEIGHTS.exists():
    logging.error(f"Model weights not found at {WEIGHTS}")
    exit(1)
if not DATA_YAML.exists():
    logging.error(f"data.yaml missing at {DATA_YAML}")
    exit(1)

# Run validation and capture DetMetrics object
logging.info("Running YOLOv8 validation and capturing metrics from API…")
det_metrics = YOLO(str(WEIGHTS)).val(
    data=str(DATA_YAML),
    save_json=False  # skip JSON file creation
)

# Extract metrics dict via results_dict property
metrics_dict = det_metrics.results_dict  # precision, recall, mAP, F1, etc.

# Log metrics
logging.info("Validation metrics:")
for name, value in metrics_dict.items():
    logging.info(f"  {name}: {value}")

# Save metrics JSON under data root logs directory
metrics_path = LOG_DIR / "val_metrics.json"
metrics_path.write_text(json.dumps(metrics_dict, indent=2))
logging.info(f"Saved summary metrics to {metrics_path}")
