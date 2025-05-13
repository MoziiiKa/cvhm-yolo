#!/usr/bin/env bash

# =============================================
# setup_and_run.sh - Setup environment and run
#   the entire CVHM-YOLO pipeline end-to-end
# =============================================

LOG_FILE="setup_and_run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================="
echo "Starting setup and runner script"
echo "Log file: $LOG_FILE"
echo "Timestamp: $(date)"
echo "============================================="

# 1) Check for Python
if ! command -v python &>/dev/null; then
  echo "🚨 ERROR: Python is not installed. Please install Python 3.8+ and try again."
  exit 1
else
  echo "✅ Python found: $(python --version)"
fi

# 2) Check for virtual environment, create if necessary
VENV_DIR="cvhm-yolo-venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "⚡️ Virtual environment not found, creating $VENV_DIR..."
  python -m venv $VENV_DIR || { echo "Failed to create virtual env"; exit 1; }
fi

# Activate venv
source $VENV_DIR/bin/activate || { echo "Failed to activate virtual env"; exit 1; }
echo "✅ Activated virtual environment: $VENV_DIR"

# 3) Install dependencies
if [ -f requirements.txt ]; then
  echo "Installing Python dependencies from requirements.txt..."
  pip install --upgrade pip setuptools wheel
  pip install -r requirements.txt || { echo "Dependency installation failed"; exit 1; }
else
  echo "⚡️ WARNING: requirements.txt not found, skipping pip install"
fi

# 4) Check Kaggle API credentials
if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  echo "🚨 ERROR: Kaggle API key not found at ~/.kaggle/kaggle.json";
  echo "Please place your kaggle.json there and run 'chmod 600 ~/.kaggle/kaggle.json'"
  exit 1
else
  chmod 600 "$HOME/.kaggle/kaggle.json"
  echo "✅ Kaggle credentials found and secured"
fi

# 5) Check .env for Roboflow API key
if [ ! -f ".env" ]; then
  echo "🚨 ERROR: .env file not found in project root."
  echo "✅ Create a .env containing ROBOFLOW_API_KEY=your_key"
  exit 1
else
  echo ".env file found"
fi

# 6) Run pipeline scripts sequentially
SCRIPTS=( 
  "scripts/download_kaggle.py" 
  "scripts/download_roboflow.py" 
  "scripts/convert_kaggle_annotations.py" 
  "scripts/prepare_dataset.py" 
  "scripts/train_yolo.py" 
)

for script in "${SCRIPTS[@]}"; do
  echo "---------------------------------------------"
  echo "🚀 Running $script"
  python $script || { echo "🚨 ERROR: $script failed. See log for details."; exit 1; }
done

echo "---------------------------------------------"
echo "🎉 Pipeline completed successfully at $(date)"
exit 0
