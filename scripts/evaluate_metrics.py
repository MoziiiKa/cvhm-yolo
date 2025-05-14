#!/usr/bin/env python3
"""
Runs a full validation pass,
saves the numeric metrics (mAP@[.5:.95],
precision, recall, F1) to JSON, and logs them
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

# Logging setup
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

# Paths to model & validation output location under data_root
WEIGHTS      = DATA_ROOT / "runs" / "exp" / "weights" / "best.pt"
VAL_PROJECT  = DATA_ROOT / "runs" / "val"
VAL_RUN      = VAL_PROJECT / "exp"
METRICS_JSON = VAL_RUN / "results.json"

# Validate required files
if not WEIGHTS.exists():
    logging.error(f"Model weights not found at {WEIGHTS}")
    exit(1)
if not (ROOT / "data.yaml").exists():
    logging.error("data.yaml missing in project root")
    exit(1)

# Run validation with JSON export into the data root
logging.info("Running YOLOv8 validation with JSON export…")
YOLO(str(WEIGHTS)).val(
    data=str(ROOT / "data.yaml"),
    save_json=True,
    project=str(VAL_PROJECT),
    name="exp",
    exist_ok=True
)

# Load & log metrics
if METRICS_JSON.exists():
    metrics = json.loads(METRICS_JSON.read_text())
    logging.info("Validation metrics:")
    for k, v in metrics.items():
        logging.info(f"  {k}: {v}")
    # Persist a summary copy at project root
    summary_path = ROOT / "val_metrics.json"
    summary_path.write_text(json.dumps(metrics, indent=2))
    logging.info(f"Saved summary JSON to {summary_path}")
else:
    logging.error(f"Expected metrics JSON not found at {METRICS_JSON}")
    exit(1)
