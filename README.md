# VEYA Fundus Inference API

FastAPI service for `best_model.pth` from the VEYA Colab bundle.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check: `GET /health`

Prediction: `POST /predict` with multipart field `file` containing a JPG or PNG fundus image.

## Docker

```bash
docker build -t veya-inference .
docker run --rm -p 8000:8000 veya-inference
```

The model is a screening aid, not a diagnosis. Validate performance clinically before use.
