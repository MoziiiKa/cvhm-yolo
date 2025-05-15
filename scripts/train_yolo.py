#!/usr/bin/env python3
"""
Train YOLOv8-nano on CPU/GPU with:
- centralized data_root for datasets
- automatic download & relocation of pretrained weights into data_root/models
- absolute path resolution for data.yaml splits
- logs written to data_root/logs/train.log
- training outputs to data_root/runs/exp/
"""
import os
import json
import yaml
import logging
import shutil
from pathlib import Path
import torch
from ultralytics import YOLO

# -------------------------------
# Load config & resolve data_root
# -------------------------------
PROJECT   = Path(__file__).resolve().parent.parent
cfg       = json.load(open(PROJECT / 'config.json'))
raw_root  = cfg['data_root']
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))

# -------------------------------
# Prepare logs directory
# -------------------------------
LOG_DIR   = DATA_ROOT / cfg.get('logs_dir', 'logs')
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE  = LOG_DIR / 'train.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)
logging.info(f"Project root: {PROJECT}")
logging.info(f"Data root: {DATA_ROOT}")

# --------------------------------------
# Preload & relocate pretrained weights
# --------------------------------------
MODELS_DIR = DATA_ROOT / cfg.get('models_dir', 'models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# List pretrained weights to ensure
required_weights = [cfg['training']['model']]
# optional extras
for extra in cfg.get('extra_models', []):
    required_weights.append(extra)

for fname in required_weights:
    dest = MODELS_DIR / fname
    if not dest.exists():
        logging.info(f"Pretrained model {fname} not found, downloading...")
        # triggers download to CWD
        _ = YOLO(fname)
        src = PROJECT / fname
        if src.exists():
            shutil.move(str(src), str(dest))
            logging.info(f"Moved {fname} to {dest}")
        else:
            logging.error(f"Downloaded file {src} not found!")
            raise FileNotFoundError(f"Could not locate pretrained weight {fname}")

# Final model path
model_path = MODELS_DIR / cfg['training']['model']
logging.info(f"Using model weights at {model_path}")

# -------------------------------
# Resolve data.yaml to abs paths
# -------------------------------
orig_yaml = PROJECT / 'data.yaml'
with open(orig_yaml) as f:
    dy = yaml.safe_load(f)
for split in ('train', 'val', 'test'):
    dy[split] = str(Path(dy[split]).expanduser().resolve())
abs_yaml = PROJECT / 'data_abs.yaml'
with open(abs_yaml, 'w') as f:
    yaml.dump(dy, f)
logging.info(f"Wrote absolute data config to {abs_yaml}")

# -------------------------------
# Select device
# -------------------------------
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

# -------------------------------
# Train
# -------------------------------
model = YOLO(str(model_path))
logging.info("Loaded YOLO model for training")
model.train(
    data=str(abs_yaml),
    epochs=cfg['training']['epochs'],
    imgsz=cfg['training']['imgsz'],
    batch=cfg['training']['batch_size'],
    device=device,
    project=str(DATA_ROOT / 'runs'),
    name='exp',
    exist_ok=True
)
logging.info("Training complete, weights and outputs saved under data_root/runs/exp")
