import os, json, shutil
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

# Load config & resolve data_root
cfg       = json.load(open(Path(__file__).parent.parent / 'config.json'))
raw_root  = cfg['data_root']
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))

SPLITS = cfg['splits']
PREP   = cfg['prepare']['kaggle_split']

# Kaggle & Roboflow source dirs
K_IMG  = DATA_ROOT / cfg['kaggle']['path'] / 'images'
K_LBL  = DATA_ROOT / cfg['kaggle']['path'] / 'labels'
RF_DIR = DATA_ROOT / cfg['roboflow']['path']

# Target splits under data_root
IMG_OUT = DATA_ROOT / SPLITS['images_dir']
LBL_OUT = DATA_ROOT / SPLITS['labels_dir']
# Clean old outputs
shutil.rmtree(IMG_OUT, ignore_errors=True)
shutil.rmtree(LBL_OUT, ignore_errors=True)

def copy_split(src_img, src_lbl, split):
    dst_i = IMG_OUT / split
    dst_l = LBL_OUT / split
    dst_i.mkdir(parents=True, exist_ok=True)
    dst_l.mkdir(parents=True, exist_ok=True)
    for img in src_img.iterdir():
        if img.suffix.lower() not in ['.jpg', '.png']: continue
        lbl = src_lbl / f"{img.stem}.txt"
        shutil.copy(img, dst_i)
        if lbl.exists():
            shutil.copy(lbl, dst_l)

# 1) Split Kaggle
imgs = list(K_IMG.glob("*.*"))
train, temp = train_test_split(imgs, test_size=PREP['test_size'],  random_state=PREP['random_state'])
val, test  = train_test_split(temp, test_size=PREP['val_size'], random_state=PREP['random_state'])
for split, group in [('train', train), ('val', val), ('test', test)]:
    copy_split(K_IMG, K_LBL, split)

# 2) Copy Roboflow
for rf in ('train', 'valid', 'test'):
    split = 'val' if rf == 'valid' else rf
    copy_split(RF_DIR / rf / 'images', RF_DIR / rf / 'labels', split)

# 3) Write data.yaml in project root (next to config.json)
data_cfg = {
    'path':      str(DATA_ROOT),
    'train':     str(IMG_OUT / 'train'),
    'val':       str(IMG_OUT / 'val'),
    'test':      str(IMG_OUT / 'test'),
    'nc':        cfg.get('nc', 1),
    'names':     cfg.get('names', ['cow'])
}
with open(Path(__file__).parent.parent / 'data.yaml', 'w') as f:
    yaml.dump(data_cfg, f)

print("Dataset prepared under", DATA_ROOT)
