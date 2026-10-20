"""VEYA Fundus Inference API.

Serves model_bundle/model.onnx with ONNX Runtime (no PyTorch at runtime), so the
service fits in ~300 MB RAM and runs on free hosting tiers.

Two bundle formats are supported (picked from model_bundle/model_config.json):
  * "veya-v2"  — VEYA v2 model (honest patient-level split, external test). The ONNX graph already
                 contains normalisation, flip-TTA and softmax/sigmoid; the config carries per-disease
                 thresholds and a "doubt zone" so the API can answer healthy / suspected / refer.
  * legacy     — the original EfficientNet-B4 export (raw logits, ImageNet normalisation here).
"""
import ctypes
import json
import os
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

import ensemble as ens
import explain

BUNDLE_DIR = Path(__file__).resolve().parent / "model_bundle"
ENSEMBLE_DIR = BUNDLE_DIR / "ensemble"
DISCLAIMER = ("Скрининговая оценка ИИ, не диагноз. Окончательное заключение даёт врач-офтальмолог. / "
              "AI screening aid, not a diagnosis.")

app = FastAPI(title="VEYA Fundus Inference API", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_session = None
_config = None
_cam_head = None
_ensemble = None


def use_ensemble() -> bool:
    return (ENSEMBLE_DIR / "ensemble.json").exists() and os.environ.get("VEYA_ENSEMBLE", "1") != "0"


def get_runtime():
    """(session, config) for the single model, or (Ensemble, config) when model_bundle/ensemble/ is present."""
    global _session, _config, _ensemble
    if use_ensemble():
        if _ensemble is None:
            _ensemble = ens.Ensemble(ENSEMBLE_DIR, crop_fundus)
            release_memory()
        return _ensemble, _ensemble.config
    if _session is None:
        with open(BUNDLE_DIR / "model_config.json", encoding="utf-8") as f:
            _config = json.load(f)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
        opts.enable_cpu_mem_arena = False  # keeps RAM low on 512 MB instances
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            str(BUNDLE_DIR / "model.onnx"), opts, providers=["CPUExecutionProvider"]
        )
    return _session, _config


def get_cam_head():
    """Head weights for Grad-CAM; None when the bundle was not prepared with tools/add_explain_outputs.py."""
    global _cam_head
    if _cam_head is None:
        runtime, _ = get_runtime()
        if isinstance(runtime, ens.Ensemble):
            path = ENSEMBLE_DIR / "explain_head.npz"
            names = {"feat", "feat_flip"} if path.exists() else set()
        else:
            names = {o.name for o in runtime.get_outputs()}
            path = BUNDLE_DIR / "explain_head.npz"
        if {"feat", "feat_flip"} <= names and path.exists():
            _cam_head = explain.CamHead(path)
        else:
            _cam_head = False
    return _cam_head or None


def release_memory():
    """Give freed activation buffers back to the OS so RSS does not creep up on a 512 MB instance."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:  # not glibc (macOS / Windows dev machines)
        pass


def is_v2(config: dict) -> bool:
    return config.get("format") in ("veya-v2", "veya-ensemble")


def model_name(config: dict) -> str:
    return config.get("model_name", "veya-fundus-efficientnet-b4")


# ----------------------------------------------------------------------------- preprocessing
def crop_fundus(image: Image.Image, size: int) -> np.ndarray:
    """Crop black borders around the retina and centre it in a square (same as VEYA v2 training)."""
    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    thr = max(10.0, float(np.percentile(gray, 5)) + 8)
    mask = gray > thr
    rows = np.where(mask.mean(1) > 0.02)[0]
    cols = np.where(mask.mean(0) > 0.02)[0]
    if len(rows) < h * 0.2 or len(cols) < w * 0.2:
        y0, y1, x0, x1 = 0, h, 0, w
    else:
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    arr = arr[y0:y1, x0:x1]
    h, w = arr.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = arr
    out = Image.fromarray(canvas).resize((size, size), Image.BOX if s >= size else Image.BICUBIC)
    return np.asarray(out, dtype=np.float32)


def preprocess(image: Image.Image, config: dict) -> np.ndarray:
    size = int(config["image_size"])
    if is_v2(config):
        x = crop_fundus(image, size)  # RGB 0..255; normalisation is inside the ONNX graph
        return x.transpose(2, 0, 1)[None, ...]
    # legacy: torchvision Resize((s, s)) + ToTensor + Normalize
    image = image.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(image, dtype=np.float32) / 255.0
    x = (x - np.array(config["mean"], dtype=np.float32)) / np.array(config["std"], dtype=np.float32)
    return x.transpose(2, 0, 1)[None, ...]


def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


# ----------------------------------------------------------------------------- decision logic (v2)
DISEASE_RU = {"dr": "Диабетическая ретинопатия", "glaucoma": "Глаукома", "cataract": "Катаракта"}
STATUS_RU = {"positive": "признаки обнаружены", "negative": "признаков нет", "uncertain": "нужна проверка врача"}


def disease_status(p: float, thr: float, zone) -> str:
    lo, hi = zone
    if lo <= p < hi:
        return "uncertain"
    return "positive" if p >= thr else "negative"


def v2_response(outputs: dict, config: dict) -> dict:
    grade = outputs["dr_grade"][0].astype(np.float64)
    grade = (grade / grade.sum()).tolist()
    probs = {k: float(outputs[k].reshape(-1)[0]) for k in ("dr", "glaucoma", "cataract")}
    thr = config["thresholds"]
    zones = config["doubt_zones"]

    screening = {}
    for k, p in probs.items():
        st = disease_status(p, thr[k], zones[k])
        screening[k] = {
            "name_ru": DISEASE_RU[k],
            "probability": round(p, 4),
            "threshold": round(float(thr[k]), 4),
            "status": st,
            "status_ru": STATUS_RU[st],
        }
    stage = int(np.argmax(grade))
    names = config.get("dr_grade_names", ["0", "1", "2", "3", "4"])
    screening["dr_stage"] = {"class": stage, "name_ru": names[stage],
                             "probabilities": [round(v, 4) for v in grade],
                             "applies": screening["dr"]["status"] != "negative"}

    statuses = [screening[k]["status"] for k in ("dr", "glaucoma", "cataract")]
    if "positive" in statuses:
        found = [DISEASE_RU[k] for k in ("dr", "glaucoma", "cataract") if screening[k]["status"] == "positive"]
        overall = {"status": "disease_suspected",
                   "message_ru": "Обнаружены признаки: " + ", ".join(found) + ". Рекомендуется консультация офтальмолога."}
    elif "uncertain" in statuses:
        overall = {"status": "refer",
                   "message_ru": "ИИ не уверен в результате. Рекомендуется проверка у офтальмолога."}
    else:
        overall = {"status": "healthy",
                   "message_ru": "Признаков диабетической ретинопатии, глаукомы и катаракты не обнаружено."}
    screening["overall"] = overall

    # backward-compatible block (same keys the legacy API returned)
    result = {
        "dr_grade": {"class": stage, "probabilities": grade, "confidence": max(grade)},
        "glaucoma": {"probability": probs["glaucoma"], "positive": screening["glaucoma"]["status"] == "positive"},
        "cataract": {"probability": probs["cataract"], "positive": screening["cataract"]["status"] == "positive"},
        "dr": {"probability": probs["dr"], "positive": screening["dr"]["status"] == "positive"},
    }
    return {"model": model_name(config), "result": result, "screening": screening, "disclaimer": DISCLAIMER}


def legacy_response(outputs: dict, config: dict) -> dict:
    result = {}
    for name, kind in config["targets"].items():
        if kind == "multiclass":
            probabilities = softmax(outputs[name][0].astype(np.float64)).tolist()
            class_index = int(np.argmax(probabilities))
            result[name] = {"class": class_index, "probabilities": probabilities, "confidence": max(probabilities)}
        else:
            probability = float(1.0 / (1.0 + np.exp(-float(outputs[name].reshape(-1)[0]))))
            result[name] = {"probability": probability, "positive": probability >= 0.5}
    return {"model": model_name(config), "result": result, "disclaimer": DISCLAIMER}


# ----------------------------------------------------------------------------- routes
@app.on_event("startup")
def preload_model():
    # Load weights at boot so the first /predict from the platform isn't slow.
    try:
        runtime, _ = get_runtime()
        if isinstance(runtime, ens.Ensemble):
            # one warm-up pass touches every model: if memory is short the boot fails here, before the health
            # check passes, so the host keeps serving the previous version instead of crashing on a patient photo
            runtime.predict(Image.new("RGB", (1024, 1024), (150, 70, 40)))
            release_memory()
        print(f"[STARTUP] Model loaded, explanations: {'on' if get_cam_head() else 'off'}")
    except Exception as exc:  # health check will report it
        print(f"[STARTUP] Model failed to load: {exc}")


@app.get("/")
def root():
    return {
        "service": "VEYA Fundus Inference API",
        "version": app.version,
        "status": "online",
        "endpoints": {"health": "/health", "docs": "/docs", "predict": "/predict?lang=ru&explain=true", "model": "/model"},
    }


@app.get("/health")
def health():
    try:
        _, config = get_runtime()
        targets = ["dr_grade", "dr", "glaucoma", "cataract"] if is_v2(config) else config["targets"]
        info = {"status": "ok", "device": "cpu", "runtime": "onnxruntime", "model": model_name(config),
                "targets": targets, "explanations": bool(get_cam_head())}
        if config.get("format") == "veya-ensemble":
            info["ensemble"] = {"models": [ens.MODEL_LABELS.get(m["key"], m["key"]) for m in config["models"]],
                                "decision": "brain"}
        return info
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@app.get("/model")
def model_info():
    """Model card: thresholds and honest test metrics (v2 bundles)."""
    _, config = get_runtime()
    keys = ("model_name", "backbone", "models", "decision", "image_size", "thresholds", "doubt_zones", "metrics")
    return {k: config[k] for k in keys if k in config}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lang: str = Query("ru", description="Language of explanation and recommendations: ru, en, kk"),
    explain_answer: bool = Query(True, alias="explain", description="Add Grad-CAM explanation and recommendations"),
    heatmap: bool = Query(True, description="Include heatmap images (base64 JPEG) in the explanation"),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or other image file")

    try:
        start_time = time.time()
        runtime, config = get_runtime()
        image = ImageOps.exif_transpose(Image.open(BytesIO(await file.read())))
        print(f"[INFERENCE] Image loaded: {image.size}, file: {file.filename}")

        members = None
        if isinstance(runtime, ens.Ensemble):
            outputs, rgb, members = runtime.predict(image)
        else:
            names = [o.name for o in runtime.get_outputs()]
            x = preprocess(image, config)
            outputs = dict(zip(names, runtime.run(names, {"image": x})))
            rgb = x[0].transpose(1, 2, 0)
        response = v2_response(outputs, config) if is_v2(config) else legacy_response(outputs, config)
        if members:
            response["ensemble"] = {"decision": "brain", "models": members}
        cam_head = get_cam_head() if is_v2(config) else None
        if explain_answer and cam_head:
            response["explanation"], response["recommendations"] = explain.explain(
                rgb, outputs, response["screening"], cam_head, lang, with_images=heatmap, members=members)
        response["inference_seconds"] = round(time.time() - start_time, 2)
        release_memory()
        print(f"[INFERENCE] {response.get('screening', {}).get('overall', response['result'])} "
              f"in {response['inference_seconds']}s")
        return response
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}") from exc
