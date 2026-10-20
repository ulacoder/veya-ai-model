FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MALLOC_ARENA_MAX=2

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py explain.py ensemble.py ./
# Single model (model.onnx) and, when exported, the full ensemble in model_bundle/ensemble/ — the server
# uses the ensemble automatically if it is there (.dockerignore keeps training leftovers out).
COPY model_bundle/ ./model_bundle/
# GitHub caps files at 100 MB, so the fp32 ViT weights are stored in parts; join them back here
RUN cd model_bundle/ensemble && if ls vit.weights.part-* >/dev/null 2>&1; then \
        cat vit.weights.part-* > vit.weights && rm vit.weights.part-*; fi && ls -la

# Hosts set PORT themselves (Render: 10000, Hugging Face Spaces: 7860).
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
