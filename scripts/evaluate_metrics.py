"""
Runs a full validation pass, 
saves the numeric metrics (mAP@[.5:.95], 
precision, recall, F1) to JSON, and logs them
"""

#!/usr/bin/env python3
import json, logging, os
from pathlib import Path
from ultralytics import YOLO

# Load config & resolve data_root
ROOT      = Path(__file__).parent.parent
cfg       = json.load(open(ROOT / "config.json"))
raw_root  = cfg["data_root"]
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))

# Logging
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

# Paths to model & val outputs under data_root
WEIGHTS   = DATA_ROOT / "runs"   / "exp" / "weights" / "best.pt"
VAL_RUN    = DATA_ROOT / "runs"  / "val" / "exp"
METRICS_JSON = VAL_RUN / "results.json"

# Validate existence
if not WEIGHTS.exists():
    logging.error(f"Model weights not found at {WEIGHTS}")
    exit(1)
if not (ROOT / "data.yaml").exists():
    logging.error("data.yaml missing in project root")
    exit(1)

# Run validation (exports JSON into VAL_RUN)
logging.info("Running YOLOv8 validation with JSON export…")
YOLO(str(WEIGHTS)).val(
    data=str(ROOT / "data.yaml"),
    save_json=True
)

# Load & log metrics
if METRICS_JSON.exists():
    metrics = json.loads(METRICS_JSON.read_text())
    logging.info("Validation metrics:")
    for k,v in metrics.items():
        logging.info(f"  {k}: {v}")
    # Save a copy at project root
    (ROOT / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    logging.info("Saved summary to val_metrics.json")
else:
    logging.error(f"Expected metrics JSON not found at {METRICS_JSON}")
    exit(1)
