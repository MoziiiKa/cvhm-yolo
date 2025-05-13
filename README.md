# CVHM-YOLO: Computer-Vision-Only Herd Monitoring System

This repository implements a **Computer-Vision-Only Herd Monitoring** prototype (similar to AIHerd) using [YOLOv8](https://github.com/ultralytics/ultralytics) on CPU/GPU. The entire pipeline—from data download, annotation conversion, dataset preparation, to model training—lives under a configurable data root (`~/data/data-cvhm-yolo/`). Everything is scripted for **one‑command reproducibility**.

---

## 📁 Repository Structure

```text
cvhm-yolo/
├─ config.json             # Pipeline configuration (paths, splits, hyperparams)
├─ .env                    # (Git-ignored) Roboflow API key
├─ requirements.txt        # Pinned Python dependencies
├─ setup_and_run.sh        # Setup & run all steps end-to-end
├─ data.yaml               # YOLOv8 dataset config (auto-rewritten)
├─ data_abs.yaml           # Absolute-path dataset config (auto-generated)
├─ scripts/
│   ├─ download_kaggle.py        # Download Kaggle Cows dataset
│   ├─ download_roboflow.py      # Download Roboflow cow project
│   ├─ convert_kaggle_annotations.py  # VOC XML → YOLO TXT converter
│   ├─ prepare_dataset.py        # Split & merge Kaggle+Roboflow into train/val/test
│   └─ train_yolo.py             # Train YOLOv8-nano on CPU/GPU with logs
└─ setup_and_run.log       # Captured output from last run
```

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
   git clone https://github.com/MoziiiKa/cvhm-yolo.git
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
