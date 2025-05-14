#!/usr/bin/env python3
"""
Hold-out Testing on the 'test' split using YOLOv8's Python API.
Extracts mAP@[.5:.95], precision, recall, and F1, then logs and saves them.
"""
import json
import logging
import os
from pathlib import Path
from ultralytics import YOLO

# 1. Load config & resolve data_root
ROOT      = Path(__file__).parent.parent
cfg       = json.load(open(ROOT / "config.json"))
raw_root  = cfg["data_root"]
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))

# 2. Setup logging to logs directory under data_root
LOG_DIR = DATA_ROOT / cfg.get("logs_dir", "logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "test_generalization.log"),
        logging.StreamHandler()
    ]
)

# 3. Paths for model and data config
WEIGHTS   = DATA_ROOT / "runs" / "exp" / "weights" / "best.pt"
DATA_YAML = ROOT / "data.yaml"

# 4. Sanity checks
if not WEIGHTS.exists():
    logging.error(f"Model weights not found at {WEIGHTS}")
    exit(1)
if not DATA_YAML.exists():
    logging.error(f"data.yaml missing at {DATA_YAML}")
    exit(1)

# 5. Run validation on the TEST split
logging.info("Running YOLOv8 hold-out test evaluation…")
det_metrics = YOLO(str(WEIGHTS)).val(
    data=str(DATA_YAML),
    split="test",      # use the test split defined in data.yaml
    save_json=False    # skip writing predictions JSON
)

# 6. Extract and log metrics
metrics = det_metrics.results_dict  # contains precision, recall, mAP, F1
logging.info("Test split metrics:")
for name, value in metrics.items():
    logging.info(f"  {name}: {value}")

# 7. Save metrics JSON for reproducibility
out_path = LOG_DIR / "test_metrics.json"
out_path.write_text(json.dumps(metrics, indent=2))
logging.info(f"Saved test metrics to {out_path}")
