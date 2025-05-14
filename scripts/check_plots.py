#!/usr/bin/env python3
import logging, os
from pathlib import Path
import json

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
        logging.FileHandler(LOG_DIR / "check_plots.log"),
        logging.StreamHandler()
    ]
)

# Validation plots location under data_root
VAL_DIR = DATA_ROOT / "runs" / "val" / "exp"
plots   = [
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "P_curve.png",
    "R_curve.png",
    "F1_curve.png"
]

found_all = True
for p in plots:
    path = VAL_DIR / p
    if path.exists():
        logging.info(f"Found {p} at {path}")
    else:
        logging.warning(f"Missing plot: {p}")
        found_all = False

if found_all:
    logging.info("All expected validation plots are present.")
else:
    logging.warning("Some validation plots are missing. Check your validation run settings.")
