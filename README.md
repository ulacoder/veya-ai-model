# VEYA Fundus Inference API

FastAPI service for the VEYA fundus model.

**Current model: VEYA v2** — EfficientNetV2-S, 512×512, multi-task: DR yes/no + DR grade 0–4, glaucoma, cataract.
Trained on 6 public datasets (EyePACS/APTOS, DDR, ODIR-5K, SMDG, AIROGS) with near-duplicate removal and a patient-level split; IDRiD and PAPILA are held out as an external test.
The server runs `model_bundle/model.onnx` with ONNX Runtime — no PyTorch at runtime, ~310 MB peak RAM.

## Honest test metrics (VEYA v2, single model)

| | AUC | Accuracy | Sensitivity | Specificity | Accuracy on confident answers (share) |
|---|---|---|---|---|---|
| DR, internal test (n=11 183) | 0.939 | 89.7% | 84.6% | 91.9% | 94.5% (75%) |
| Glaucoma, internal test (n=2 036) | 0.972 | 91.1% | 91.8% | 90.9% | 96.2% (88%) |
| Cataract, internal test (n=730) | 0.964 | 98.9% | 85.3% | 99.6% | 98.9% (100%) |
| DR, external IDRiD (n=510) | 0.973 | 85.3% | 98.8% | 57.7% | 88.8% (89%) |
| Glaucoma, external PAPILA (n=420) | 0.820 | 32.4% | 95.4% | 15.9% | 54.7% (25%) |

DR grade 0–4 (internal): accuracy 84.2%, quadratic kappa 0.847. Glaucoma does not yet transfer well to PAPILA
(over-calls glaucoma on that camera) — treat glaucoma output as "refer for check", not a diagnosis.
Thresholds and the "doubt zone" (answers inside it are returned as `uncertain` → refer to a doctor) are in
`model_bundle/model_config.json` and served at `GET /model`.

## Endpoints

- `GET /health` — model status
- `POST /predict` — multipart field `file` with a JPG/PNG fundus image. Response:
  - `result` — same keys as v1 (`dr_grade`, `glaucoma`, `cataract`) plus `dr`;
  - `screening` — per disease `probability`, `threshold`, `status` (`positive` / `negative` / `uncertain`) with Russian labels,
    `dr_stage`, and `overall.status` (`healthy` / `disease_suspected` / `refer`) with a Russian message.
- `GET /model` — thresholds and test metrics
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

VEYA v2 models are trained and exported in Google Colab (notebooks `VEYA_02_train_models` and `VEYA_05_export_onnx`).
The export writes `model.onnx` (normalisation + flip-TTA + softmax/sigmoid inside the graph) and `model_config.json`
to `MyDrive/veya_v2/export/`; copy both into `model_bundle/`. `best_model.pth`, `inference.py` and `tools/export_onnx.py`
belong to the previous EfficientNet-B4 model; `main.py` still serves that format if its config is restored.

The model is a screening aid, not a diagnosis. Validate performance clinically before use.
