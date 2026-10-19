"""VEYA v2 ensemble at inference time: three base models give their opinions, the brain decides.

Files come from tools/export_ensemble.py (model_bundle/ensemble/). Each base model sees the fundus cropped
and resized to its own input size; the brain (NumPy port of training/veya_v2/brain.py) takes every model's
DR-stage probabilities, disease logits and PCA-compressed embedding and returns the final probabilities.
The final answer is always the brain's.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_LABELS = {"effv2s": "EfficientNetV2-S", "convnext": "ConvNeXt-Small", "vit": "ViT-B/14 DINOv2"}


def _softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-v))


_erf = np.vectorize(math.erf)


def _gelu(v: np.ndarray) -> np.ndarray:
    return 0.5 * v * (1.0 + _erf(v / math.sqrt(2.0)))


class Brain:
    def __init__(self, path: Path, models: list[str]):
        w = np.load(path)
        self.models = models
        self.mu, self.sd = w["mu"], w["sd"]
        self.pca = {m: (w[f"pca_{m}_components"], w[f"pca_{m}_mean"]) for m in models}
        self.nets = []
        for i in range(int(w["n_nets"])):
            layers = [(w[f"net{i}_w{j}"], w[f"net{i}_b{j}"]) for j in range(3)]
            self.nets.append((layers, (w[f"net{i}_grade_w"], w[f"net{i}_grade_b"]), (w[f"net{i}_bin_w"], w[f"net{i}_bin_b"])))

    def __call__(self, opinions: dict) -> tuple[np.ndarray, np.ndarray]:
        """opinions[model] = dict(grade_logits [5], bin_logits [3], emb [D]) → (grade probs [5], disease probs [3])."""
        cols = []
        for m in self.models:
            o = opinions[m]
            comps, mean = self.pca[m]
            cols += [_softmax(o["grade_logits"]), o["bin_logits"], (o["emb"] - mean) @ comps.T]
        x = (np.concatenate(cols).astype(np.float32) - self.mu) / self.sd
        base = np.concatenate([np.mean([opinions[m]["grade_logits"] for m in self.models], 0),
                               np.mean([opinions[m]["bin_logits"] for m in self.models], 0)])
        grades, bins = [], []
        for layers, (gw, gb), (bw, bb) in self.nets:
            h = x
            for w, b in layers:
                h = _gelu(w @ h + b)
            grades.append(_softmax(gw @ h + gb + base[:5]))
            bins.append(_sigmoid(bw @ h + bb + base[5:]))
        return np.mean(grades, 0), np.mean(bins, 0)


class Ensemble:
    def __init__(self, directory: Path, crop):
        self.dir = Path(directory)
        self.config = json.loads((self.dir / "ensemble.json").read_text(encoding="utf-8"))
        self.crop = crop
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "1"))
        opts.enable_cpu_mem_arena = False
        opts.add_session_config_entry("session.disable_prepacking", "1")  # no packed weight copies: −60 MB peak
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.members = []
        for m in self.config["models"]:
            sess = ort.InferenceSession(str(self.dir / m["file"]), opts, providers=["CPUExecutionProvider"])
            self.members.append((m["key"], int(m["size"]), sess, [o.name for o in sess.get_outputs()]))
        self.keys = [k for k, *_ in self.members]
        self.brain = Brain(self.dir / self.config["brain"]["file"], self.keys)

    def predict(self, image: Image.Image) -> tuple[dict, np.ndarray, dict]:
        """→ (outputs in the single-model layout + feat maps of the CAM model, CAM model input RGB, per-model opinions)."""
        opinions, cam_out, cam_rgb = {}, None, None
        for key, size, sess, names in self.members:
            rgb = self.crop(image, size)
            out = dict(zip(names, sess.run(names, {"image": rgb.transpose(2, 0, 1)[None]})))
            opinions[key] = {"grade_logits": out["grade_logits"][0].astype(np.float64),
                             "bin_logits": out["bin_logits"][0].astype(np.float64),
                             "emb": out["emb"][0].astype(np.float32),
                             "probs": {"dr": float(out["dr"][0]), "glaucoma": float(out["glaucoma"][0]),
                                       "cataract": float(out["cataract"][0])}}
            if key == self.config["cam_model"]:
                cam_out, cam_rgb = out, rgb
        grade, bins = self.brain(opinions)
        outputs = {"dr_grade": grade[None], "dr": bins[0:1], "glaucoma": bins[1:2], "cataract": bins[2:3]}
        if cam_out is not None and "feat" in cam_out:
            outputs["feat"], outputs["feat_flip"] = cam_out["feat"], cam_out["feat_flip"]
        members = {k: {"name": MODEL_LABELS.get(k, k), **o["probs"]} for k, o in opinions.items()}
        return outputs, cam_rgb, members
