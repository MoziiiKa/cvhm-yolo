# deploy/app.py
import io
import os
from pathlib import Path
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

# Load config
ROOT      = Path(__file__).parent.parent
import json
cfg       = json.load(open(ROOT / "config.json"))
raw_root  = cfg["data_root"]
DATA_ROOT = Path(os.path.expanduser(os.path.expandvars(raw_root)))      

# Load the trained model
MODEL_PATH = DATA_ROOT / "runs/exp/weights/best.pt"
model = YOLO(str(MODEL_PATH))  # Ultralytics API

app = FastAPI(title="CVHM-YOLO Inference API")

@app.post("/detect-image/")
async def detect_image(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # Run inference
    results = model(img)[0]  # returns a Results object
    # Render annotated image
    annotated = results.plot()
    _, img_encoded = cv2.imencode('.jpg', annotated)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@app.post("/detect-video/")
async def detect_video(file: UploadFile = File(...)):
    tmp_in = Path(tempfile.mkdtemp()) / file.filename
    with open(tmp_in, "wb") as f: f.write(await file.read())
    cap = cv2.VideoCapture(str(tmp_in))
    # Create temp output
    tmp_out = Path(tempfile.mkdtemp()) / ("out_" + file.filename)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_out), fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    while True:
        ret, frame = cap.read()
        if not ret: break
        res = model(frame)[0]
        out.write(res.plot())
    cap.release(); out.release()
    # return StreamingResponse(open(tmp_out, "rb"), media_type="video/mp4")
    return FileResponse(
        path=str(tmp_out),
        media_type="video/mp4",
        filename=f"out_{file.filename}"
    )
