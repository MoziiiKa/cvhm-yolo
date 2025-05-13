import os, json
from pathlib import Path
import xml.etree.ElementTree as ET

# Load config & resolve data_root
cfg       = json.load(open(Path(__file__).parent.parent / 'config.json'))
raw_root  = cfg['data_root']
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))

# Paths under data_root
XML     = DATA_ROOT / cfg['kaggle']['path'] / 'annotations.xml'
IMG_DIR = DATA_ROOT / cfg['kaggle']['path'] / 'images'
LBL_DIR = DATA_ROOT / cfg['kaggle']['path'] / 'labels'
LBL_DIR.mkdir(parents=True, exist_ok=True)

# Parse and convert
root = ET.parse(XML).getroot()
def voc2y(xtl, ytl, xbr, ybr, w,h): return ((xtl+xbr)/2)/w,((ytl+ybr)/2)/h,(xbr-xtl)/w,(ybr-ytl)/h
for img in root.findall('image'):
    name = Path(img.get('name')).name
    w,h = float(img.get('width')), float(img.get('height'))
    lines=[]
    for box in img.findall('box'):
        if box.get('label')!='cow': continue
        xt,yt, xb,yb = map(float,[box.get('xtl'),box.get('ytl'),box.get('xbr'),box.get('ybr')])
        xc,yc,wd,ht = voc2y(xt,yt,xb,yb,w,h)
        lines.append(f"0 {xc:.6f} {yc:.6f} {wd:.6f} {ht:.6f}")
    if lines:
        (LBL_DIR / f"{Path(name).stem}.txt").write_text("\n".join(lines))
print("Converted Kaggle XML to YOLO labels")