from pathlib import Path
import sys

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

BUNDLE_DIR = Path(__file__).resolve().parent / "model_bundle"
sys.path.insert(0, str(BUNDLE_DIR))

try:
    from inference import load_model
except Exception as exc:  # pragma: no cover
    raise RuntimeError(f"Unable to import model bundle: {exc}") from exc

app = FastAPI(title="VEYA Fundus Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_model = None
_config = None
_device = None


@app.get("/")
def root():
    return {"service": "VEYA Fundus Inference API", "status": "online", "health": "/health", "docs": "/docs", "predict": "/predict"}


@app.get("/health")
def health():
    global _model, _config, _device
    if _model is None:
        _model, _config, _device = load_model(str(BUNDLE_DIR))
    return _model, _config, _device


@app.get("/health")
def health():
    try:
        _, config, device = get_runtime()
        return {"status": "ok", "device": str(device), "targets": config["targets"]}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {exc}") from exc


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or other image file")

    try:
        from io import BytesIO
        from PIL import Image
        import torch
        from torchvision import transforms

        model, config, device = get_runtime()
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((config["image_size"], config["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(config["mean"], config["std"]),
            ]
        )
        with torch.inference_mode():
            outputs = model(transform(image).unsqueeze(0).to(device))

        result = {}
        for name, kind in config["targets"].items():
            if kind == "multiclass":
                probabilities = torch.softmax(outputs[name], dim=1)[0].detach().cpu().tolist()
                class_index = int(max(range(len(probabilities)), key=probabilities.__getitem__))
                result[name] = {
                    "class": class_index,
                    "probabilities": probabilities,
                    "confidence": max(probabilities),
                }
            else:
                probability = float(torch.sigmoid(outputs[name].squeeze()).detach().cpu())
                result[name] = {"probability": probability, "positive": probability >= 0.5}

        return {"model": "veya-fundus-efficientnet-b4", "result": result}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Inference failed: {exc}") from exc
