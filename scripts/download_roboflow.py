import os, json
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv

# Load config & env & resolve data_root
cfg       = json.load(open(Path(__file__).parent.parent / 'config.json'))
load_dotenv()
raw_root  = cfg['data_root']
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))

# Roboflow download path
RF_PATH = DATA_ROOT / cfg['roboflow']['path']
RF_PATH.mkdir(parents=True, exist_ok=True)

# os.makedirs(RF_PATH, exist_ok=True)
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY not found; ensure .env is configured")
rf = Roboflow(api_key=api_key)

project = rf.workspace(cfg['roboflow']['workspace']).project(cfg['roboflow']['project'])
dataset = project.version(cfg['roboflow']['version']).download(
    model_format="yolov8",
    location=str(RF_PATH),
    overwrite=True
)

print("Files extracted to: ", dataset.location)