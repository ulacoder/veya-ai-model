# VEYA Fundus Inference API

FastAPI service for the VEYA fundus model (EfficientNet-B4, 512×512; targets: DR grade 0–4, glaucoma, cataract).
The server runs `model_bundle/model.onnx` with ONNX Runtime — no PyTorch at runtime, ~310 MB peak RAM.

## Endpoints

- `GET /health` — model status
- `POST /predict` — multipart field `file` with a JPG/PNG fundus image
- `GET /docs` — interactive API docs

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t veya-inference .
docker run --rm -p 8000:8000 veya-inference
```

## Deploy

- **Render** (free plan): New → Blueprint → this repo. `render.yaml` sets everything up and redeploys on every push to `master`.
- **Hugging Face Spaces**: manual workflow `Deploy to Hugging Face Space` (needs `HF_TOKEN` secret; Docker Spaces need a PRO plan).

## After retraining

`best_model.pth` is the training checkpoint; the server only uses `model.onnx`. Re-export it:

```bash
pip install -r tools/requirements-export.txt
python tools/export_onnx.py
```

The model is a screening aid, not a diagnosis. Validate performance clinically before use.
