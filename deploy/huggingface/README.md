---
title: VEYA Fundus Inference
emoji: 👁️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 8000
pinned: false
short_description: Fundus screening API — DR, glaucoma, cataract
---

# VEYA Fundus Inference API

FastAPI service (EfficientNet-B4) used by the VEYA AI screening platform.

- `GET /health` — model status
- `POST /predict` — multipart field `file` with a JPG/PNG fundus image
- `GET /docs` — interactive API docs

Deployed automatically from GitHub (`ulacoder/veya-inference`, branch `master`).

The model is a screening aid, not a diagnosis. Clinical validation is required before use.
