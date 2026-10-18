"""VEYA v2 — экспорт обученной модели в ONNX для API (ONNX Runtime, без PyTorch на сервере).

В ONNX зашито: нормализация + TTA (обычный и зеркальный снимок, логиты усредняются) + softmax/sigmoid.
Вход:  image — float32 [1, 3, S, S], RGB, значения 0..255, снимок уже обрезан по кругу сетчатки (crop_fundus).
Выходы: dr_grade [1,5] (вероятности стадий 0–4), dr / glaucoma / cataract [1] (вероятности).

Рядом сохраняется model_config.json: пороги и «зона сомнения» по каждой болезни (подобраны на val),
и честные метрики на test / test_ext — чтобы API отвечал «здоров / болен / нужен врач».
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from . import brain as B
from . import train as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
OUTPUTS = ["dr_grade", "dr", "glaucoma", "cataract"]


class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) * 255)

    def forward(self, image):
        x = (image - self.mean) / self.std
        a = self.model(x)
        b = self.model(torch.flip(x, dims=[3]))
        g = torch.softmax((a["dr_grade"] + b["dr_grade"]) / 2, dim=1)
        return (g,) + tuple(torch.sigmoid((a[k] + b[k]) / 2) for k in ("dr", "glaucoma", "cataract"))


def calibrate(key, drive_dir, labels: pd.DataFrame):
    """Пороги и зоны сомнения на val (TTA-логиты из export_predictions) + метрики на test/test_ext."""
    lab = labels.set_index("image_id")
    P = dict(np.load(Path(drive_dir) / "preds" / f"{key}_val.npz", allow_pickle=True))
    yv = lab.loc[P["image_id"]]
    thresholds, zones = {}, {}
    for k in B.BINARY:
        y = yv[k].values
        m = y >= 0
        p = B.sigmoid(P[k][m].astype(np.float64))
        thresholds[k] = B.best_threshold(y[m], p)
        zones[k] = B.doubt_zone(y[m], p, thresholds[k])
    metrics = {}
    for sp in ["test", "test_ext"]:
        f = Path(drive_dir) / "preds" / f"{key}_{sp}.npz"
        if not f.exists():
            continue
        Q = dict(np.load(f, allow_pickle=True))
        ys = lab.loc[Q["image_id"]]
        rep = {}
        for k in B.BINARY:
            r = B.disease_report(ys[k].values, B.sigmoid(Q[k].astype(np.float64)), thresholds[k], zones[k])
            if r:
                rep[k] = {n: round(float(v), 4) for n, v in r.items()}
        mg = ys["dr_grade"].values >= 0
        if mg.sum() > 10:
            from sklearn.metrics import accuracy_score, cohen_kappa_score
            pg = Q["dr_grade"][mg].argmax(1)
            rep["dr_grade"] = dict(n=int(mg.sum()), accuracy=round(float(accuracy_score(ys["dr_grade"].values[mg], pg)), 4),
                                   qwk=round(float(cohen_kappa_score(ys["dr_grade"].values[mg], pg, weights="quadratic")), 4))
        metrics[sp] = rep
    return thresholds, zones, metrics


def export_onnx(key="effv2s", data_dir="/content/veya_data", drive_dir="/content/drive/MyDrive/veya_v2",
                out_dir=None, opset=17, check_images=8):
    out_dir = Path(out_dir or Path(drive_dir) / "export")
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(Path(drive_dir) / "models" / key / "best.pt", map_location="cpu", weights_only=False)
    size = int(ck["cfg"]["size"])
    model = T.build_model(key, pretrained=False)
    model.load_state_dict(ck["model"])
    wrapper = OnnxWrapper(model).eval()

    onnx_path = out_dir / "model.onnx"
    dummy = torch.rand(1, 3, size, size) * 255
    torch.onnx.export(wrapper, dummy, str(onnx_path), input_names=["image"], output_names=OUTPUTS,
                      opset_version=opset, dynamo=False)
    print(f"ONNX сохранён: {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} МБ)")

    labels = pd.read_csv(Path(data_dir) / "labels.csv") if (Path(data_dir) / "labels.csv").exists() \
        else pd.read_csv(Path(drive_dir) / "data" / "labels.csv")
    thresholds, zones, metrics = calibrate(key, drive_dir, labels)

    # --- проверка: ONNX Runtime == PyTorch на настоящих снимках
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    cand = labels[labels.split.isin(["test", "test_ext"])]
    test = cand[[(Path(data_dir) / p).exists() for p in cand.path]].head(check_images)
    max_diff = 0.0
    for p in test.path:
        img = T.load_image(str(Path(data_dir) / p), size).astype(np.float32)
        x = img.transpose(2, 0, 1)[None]
        o = sess.run(OUTPUTS, {"image": x})
        with torch.no_grad():
            t = wrapper(torch.from_numpy(x))
        max_diff = max(max_diff, max(float(np.abs(a - b.numpy()).max()) for a, b in zip(o, t)))
    print(f"ONNX vs PyTorch на {len(test)} снимках: макс. расхождение вероятностей {max_diff:.2e}")

    config = {
        "format": "veya-v2",
        "model_name": f"veya-v2-{key}",
        "backbone": T.MODELS[key]["timm"],
        "image_size": size,
        "input": "RGB float32 0..255, NCHW, after crop_fundus (black borders cropped, centered square)",
        "outputs": {"dr_grade": "probs[5]", "dr": "prob", "glaucoma": "prob", "cataract": "prob"},
        "dr_grade_names": ["нет ДР", "лёгкая", "умеренная", "тяжёлая", "пролиферативная"],
        "thresholds": thresholds,
        "doubt_zones": zones,
        "metrics": metrics,
        "trained_metrics_val": {k: (round(float(v), 4) if isinstance(v, (int, float)) else v)
                                for k, v in ck.get("metrics", {}).items()},
        "onnx_vs_torch_max_diff": max_diff,
    }
    (out_dir / "model_config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False))
    print(json.dumps({k: config[k] for k in ("thresholds", "doubt_zones")}, indent=1))
    for sp, rep in metrics.items():
        print(sp, {k: (v.get("auc"), v.get("accuracy")) if "auc" in v else v for k, v in rep.items()})
    print("Готово:", out_dir)
    return out_dir
