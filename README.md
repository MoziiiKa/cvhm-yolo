# CVHM-YOLO: Computer-Vision-Only Herd Monitoring System

This repository implements a **Computer-Vision-Only Herd Monitoring** solution using Ultralytics' [YOLOv8](https://github.com/ultralytics/ultralytics) on CPU/GPU. It provides an end-to-end, reproducible pipeline—from data acquisition through model training to real-time inference—organized under a configurable data root (`~/data/data-cvhm-yolo/`).

Key features:

* **One‑command setup & run**: Automated scripts download data, prepare datasets, convert annotations, train the model, and evaluate performance.
* **Modular configuration**: All paths, hyperparameters, and service settings live in `config.json`.
* **Containerized inference**: A FastAPI-based inference API with Docker support, embedding trained weights for zero‑touch deployment.
* **Comprehensive evaluation**: Scripts for quantitative validation, hold-out testing, and automated interpretation of metrics.

This repository implements a **Computer-Vision-Only Herd Monitoring** prototype (similar to AIHerd) using [YOLOv8](https://github.com/ultralytics/ultralytics) on CPU/GPU. The entire pipeline—from data download, annotation conversion, dataset preparation, to model training—lives under a configurable data root (`~/data/data-cvhm-yolo/`). Everything is scripted for **one‑command reproducibility**.

---

## 📁 Repository Structure

```text
cvhm-yolo/
├─ cvhm-yolo-venv/            # Python virtual environment
├─ deploy/                    # Containerized inference API
│   ├─ app.py                 # FastAPI application for real-time detection
│   └─ Dockerfile             # Inference image build instructions
├─ models/                    # Bundled pretrained and fine-tuned weights
│   └─ best.pt                # Trained YOLOv8 weights for inference
├─ scripts/                   # End-to-end pipeline scripts
│   ├─ download_kaggle.py     # Download Kaggle cows dataset
│   ├─ download_roboflow.py   # Download Roboflow cow detection project
│   ├─ convert_kaggle_annotations.py  # VOC XML → YOLO .txt converter
│   ├─ prepare_dataset.py     # Split & merge Kaggle + Roboflow data
│   ├─ train_yolo.py          # Train YOLOv8-nano with centralized paths
│   ├─ evaluate_metrics.py    # Validate on val split and log metrics
│   ├─ check_plots.py         # Verify confusion matrices and PR/F1 curves
│   ├─ test_generalization.py # Hold-out test split evaluation
│   └─ interpret_test_metrics.py  # Automatically interpret test metrics
├─ config.json                # Central pipeline configuration
├─ .env                       # (Git-ignored) API keys for Kaggle & Roboflow
├─ data.yaml                  # YOLOv8 dataset config (auto-updated)
├─ data_abs.yaml              # Absolute-path dataset config (auto-generated)
├─ requirements.in            # Full dependency spec (dev + prod)
├─ requirements.txt           # Pinned dependencies for local dev
├─ requirements-runtime.txt   # Slim dependencies for inference image
├─ setup_and_run.sh           # One-step setup & pipeline runner
├─ README.md                  # This documentation
└─ .gitignore                 # Ignore raw data, logs, venv, etc.
```

---

## 🚢 Deployment & Inference

Once your model is trained and `best.pt` is in `models/`, you can run the FastAPI server locally without Docker:

```bash
# Activate your virtual environment
source cvhm-yolo-venv/bin/activate

# Launch the inference API (reload for dev)
uvicorn deploy.app:app --host 0.0.0.0 --port 8889 --reload
```

* **Swagger UI**:  [http://127.0.0.1:8889/docs](http://127.0.0.1:8889/docs)
* **Redoc**:       [http://127.0.0.1:8889/redoc](http://127.0.0.1:8889/redoc)

### Endpoints

| Method | Path             | Description                                                     |
| ------ | ---------------- | --------------------------------------------------------------- |
| POST   | `/detect-image/` | Upload an image (`.jpg`, `.png`) and receive an annotated JPEG. |
| POST   | `/detect-video/` | Upload a video (`.mp4`) and receive an annotated MP4.           |

### Example: Image Detection

```bash
curl -X POST "http://127.0.0.1:8889/detect-image/" \
     -F "file=@/path/to/cow.jpg" \
     --output out.jpg
```

### Example: Video Detection

```bash
curl -X POST "http://127.0.0.1:8889/detect-video/" \
     -F "file=@/path/to/video.mp4" \
     --output out.mp4
```

You can now share your Docker image or local server command with collaborators so they can immediately test real-time detections!


---

## 🔧 Prerequisites

* Linux/macOS (Bash shell)
* Python 3.8+
* Docker (optional, not required)
* Kaggle account with API token
* Roboflow account with private API key

---

## 🚀 Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-org>/cvhm-yolo.git
   cd cvhm-yolo
   ```

2. **Configure environment secrets**

   * Place `kaggle.json` in `~/.kaggle/` and secure it:

     ```bash
     mkdir -p ~/.kaggle
     mv <downloaded>/kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```
   * Create a `.env` file in the repo root with:

     ```ini
     ROBOFLOW_API_KEY=your_private_roboflow_key
     ```

3. **Install & Activate Virtual Environment**

   ```bash
   bash setup_and_run.sh
   ```

   This script will:

   * Create/activate `cvhm-yolo-venv`
   * Install all dependencies
   * Verify credentials
   * Download datasets
   * Convert annotations
   * Prepare train/val/test splits
   * Train the YOLOv8-nano model

4. **Inspect Results**

   * Trained weights and logs land under your data root: `~/data/data-cvhm-yolo/training/weights/`
   * TensorBoard metrics and loss curves in the same folder.

---

## 📜 Configuration

All paths, dataset IDs, split ratios, and training hyperparameters live in **`config.json`**. Example:

```json
{
  "data_root": "~/data/data-cvhm-yolo",
  "kaggle": { "dataset": "trainingdatapro/cows-detection-dataset", "path": "raw/kaggle" },
  "roboflow": { "workspace": "<ws>", "project": "<proj>", "version": 1, "path": "raw/roboflow" },
  "splits": { "images_dir": "images", "labels_dir": "labels" },
  "prepare": { "kaggle_split": { "test_size": 0.3, "val_size": 0.5, "random_state": 42 } },
  "training": { "model": "yolov8n.pt", "epochs": 50, "imgsz": 640, "batch_size": 4, "device_preference": ["cuda:0","cpu"] }
}
```

Modify values as needed before running.

---

## 🔍 Pipeline Steps

1. **download\_kaggle.py**  – Fetches the Kaggle Cows dataset.
2. **download\_roboflow\.py** – Fetches the Roboflow cow detection project.
3. **convert\_kaggle\_annotations.py** – Converts PASCAL VOC XMLs to YOLO `.txt` labels.
4. **prepare\_dataset.py** – Splits Kaggle data, merges with Roboflow splits into `train/`, `val/`, `test/`.
5. **train\_yolo.py** – Trains YOLOv8-nano, auto-selects GPU/CPU, logs to `train.log`, and saves weights under `<data_root>/training/weights/`.

---

## 📝 Notes

* **Reproducibility**: All versions are pinned in `requirements.txt` and paths are driven by `config.json`.
* **Re-running**: To start fresh, delete `~/data/data-cvhm-yolo/*` and `cvhm-yolo-venv/` then re-run `setup_and_run.sh`.
* **Extensibility**: Swap `yolov8n.pt` for larger models (e.g., `yolov8s.pt`) by editing `config.json`.

---

## 📄 License



---
