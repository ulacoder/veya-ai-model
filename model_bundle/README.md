# VEYA Model Bundle

**Files:** `best_model.pth`, `model_config.json`, `inference.py`, `requirements.txt`

**Usage:**
```python
from inference import predict
result = predict("fundus.jpg", ".")
print(result)
```

**Disclaimer:** This model is a screening aid, not a diagnostic tool. Clinical validation required before deployment.