import os, json, yaml, logging
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.utils.files import WorkingDirectory

# Load config & resolve data_root
cfg       = json.load(open(Path(__file__).parent.parent / 'config.json'))
raw_root  = cfg['data_root']
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))

# Project root & logging
PROJECT   = Path(__file__).parent.parent
LOG_FILE  = PROJECT / 'train.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# Resolve data.yaml → absolute paths

# 1. Read the (already absolute) data.yaml
orig_yaml = PROJECT / 'data.yaml'
with open(orig_yaml) as f: 
    dy = yaml.safe_load(f)

# 2. Clean up each split path:
for split in ('train', 'val', 'test'):
    dy[split] = str(Path(dy[split]).expanduser().resolve())

# 3. (Optional) Write out to data_abs.yaml, or skip and use data.yaml directly
abs_yaml = PROJECT / 'data_abs.yaml'
with open(abs_yaml, 'w') as f:
    yaml.dump(dy, f)

# Select device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

with WorkingDirectory(PROJECT):
    model = YOLO(cfg['training']['model'])
    logging.info("Loaded YOLOv8 model")
    project_dir = DATA_ROOT  # ~/data/data-cvhm-yolo
    model.train(
        data=str(abs_yaml),
        epochs=cfg['training']['epochs'],
        imgsz=cfg['training']['imgsz'],
        batch=cfg['training']['batch_size'],
        device=device,
        project=str(DATA_ROOT / "runs"),      # root for runs/
        name="exp",               # subfolder name, e.g. ~/data/.../exp
        exist_ok=True                  # overwrite if the folder already exists
    )
    logging.info("Training complete, weights saved in runs/")
