"""VEYA Fundus Inference API.

Serves model_bundle/model.onnx with ONNX Runtime (no PyTorch at runtime), so the
service fits in ~300 MB RAM and runs on free hosting tiers. The ONNX file is
exported from best_model.pth with `python tools/export_onnx.py`.
"""
import json
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

BUNDLE_DIR = Path(__file__).resolve().parent / "model_bundle"
MODEL_NAME = "veya-fundus-efficientnet-b4"

app = FastAPI(title="VEYA Fundus Inference API", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_session = None
_config = None


def get_runtime():
    global _session, _config
    if _session is None:
        with open(BUNDLE_DIR / "model_config.json") as f:
            _config = json.load(f)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
        opts.enable_cpu_mem_arena = False  # keeps RAM low on 512 MB instances
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            str(BUNDLE_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
    return _session, _config


def preprocess(image: Image.Image, config: dict) -> np.ndarray:
    # Same as torchvision Resize((s, s)) + ToTensor + Normalize on a PIL image.
    size = int(config["image_size"])
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - np.array(config["mean"], dtype=np.float32)) / np.array(config["std"], dtype=np.float32)
    return x.transpose(2, 0, 1)[None, ...]


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


@app.on_event("startup")
def preload_model():
    # Load weights at boot so the first /predict from the platform isn't slow.
    try:
        get_runtime()
        print("[STARTUP] Model loaded")
    except Exception as exc:  # health check will report it
        print(f"[STARTUP] Model failed to load: {exc}")


@app.get("/")
def root():
    return {
        "service": "VEYA Fundus Inference API",
        "version": app.version,
        "status": "online",
        "endpoints": {"health": "/health", "docs": "/docs", "predict": "/predict"},
    }


@app.get("/health")
def health():
    try:
        _, config = get_runtime()
        return {"status": "ok", "device": "cpu", "runtime": "onnxruntime", "targets": config["targets"]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or other image file")

    try:
        start_time = time.time()
        session, config = get_runtime()
        image = Image.open(BytesIO(await file.read()))
        print(f"[INFERENCE] Image loaded: {image.size}, file: {file.filename}")

        names = list(config["targets"])
        outputs = dict(zip(names, session.run(names, {"image": preprocess(image, config)})))
        print(f"[INFERENCE] Inference took {time.time() - start_time:.2f}s")

        result = {}
        for name, kind in config["targets"].items():
            if kind == "multiclass":
                probabilities = softmax(outputs[name][0].astype(np.float64)).tolist()
                class_index = int(np.argmax(probabilities))
                result[name] = {
                    "class": class_index,
                    "probabilities": probabilities,
                    "confidence": max(probabilities),
                }
            else:
                probability = float(1.0 / (1.0 + np.exp(-float(outputs[name].reshape(-1)[0]))))
                result[name] = {"probability": probability, "positive": probability >= 0.5}
            print(f"[INFERENCE] {name}: {result[name]}")

        return {"model": MODEL_NAME, "result": result}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}") from exc
