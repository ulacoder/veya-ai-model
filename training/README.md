# VEYA v2 — training code

Notebooks (Google Colab unless noted):

| Notebook | What it does |
|---|---|
| `01_prepare_data` | Downloads 6 public fundus datasets (Kaggle), crops, removes near-duplicates, patient-level split, external test (IDRiD, PAPILA/RIM-ONE) |
| `02_train_models` | Trains the base models (EfficientNetV2-S, ConvNeXt-S, ViT-B/14 DINOv2), checkpoints to Google Drive |
| `03_brain_ensemble` | Trains the MLP "brain" over the models' outputs, thresholds and doubt zone, final report |
| `04_visuals` | Grad-CAM heatmaps, UMAP map of images, ROC curves |
| `05_export_onnx` | Exports a model to ONNX + `model_config.json` for the API |
| `K1_kaggle_convnext_vit` | Kaggle (2×T4): rebuilds the same split and trains ConvNeXt and ViT in parallel |

The Python modules in `veya_v2/` are the same code the notebooks write with `%%writefile`.
