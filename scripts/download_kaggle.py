import os, json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Load config & resolve data_root
cfg       = json.load(open(Path(__file__).parent.parent / 'config.json'))
raw_root  = cfg['data_root']
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))

# Kaggle download path
K_PATH = DATA_ROOT / cfg['kaggle']['path']
K_PATH.mkdir(parents=True, exist_ok=True)

os.makedirs(K_PATH, exist_ok=True)

api = KaggleApi()
api.authenticate()
print("Kaggle authentication successful!")
api.dataset_download_files(
    cfg['kaggle']['dataset'],
    path=K_PATH,
    unzip=True
)
print(f"Kaggle data downloaded to {K_PATH}")
