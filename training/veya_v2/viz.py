"""VEYA v2 — визуализации для презентации: тепловые карты (Grad-CAM), карта снимков, ROC-кривые.

Всё работает и без GPU (на CPU чуть медленнее). Картинки сохраняются в MyDrive/veya_v2/viz/.
"""
from __future__ import annotations

import os
import tarfile
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

matplotlib.rcParams["font.family"] = "DejaVu Sans"

BINARY = ["dr", "glaucoma", "cataract"]
NAMES_RU = {"dr": "Диабетическая ретинопатия", "glaucoma": "Глаукома", "cataract": "Катаракта"}
GRADE_RU = ["нет ДР", "ДР 1 (лёгкая)", "ДР 2 (умеренная)", "ДР 3 (тяжёлая)", "ДР 4 (пролиферативная)"]
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255

BG = "#0E1116"
FG = "#E8EAED"
ACCENT = {"healthy": "#5DCAA5", "dr": "#EF9F27", "glaucoma": "#7F77DD", "cataract": "#D4537E"}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ============================================================================ data helpers
def extract_some(drive_dir, local_dir, parts=("idrid", "odir", "smdg", "airogs")):
    """Распаковывает только небольшие датасеты (быстро) — для примеров тепловых карт."""
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    for p in parts:
        flag = local / f".{p}.done"
        t = Path(drive_dir) / "data" / "parts" / f"{p}.tar"
        if flag.exists() or not t.exists():
            continue
        with tarfile.open(t) as tar:
            tar.extractall(local)
        flag.touch()
        print("распакован", t.name)
    return local


def pick_examples(labels: pd.DataFrame, preds: dict, local_dir) -> list[dict]:
    """Уверенные и ПРАВИЛЬНЫЕ примеры по каждому классу (для слайдов) из доступных снимков."""
    have = {p for p in labels["path"] if (Path(local_dir) / p).exists()}
    out = []
    df = labels.set_index("image_id")
    for split, P in preds.items():
        ids = P["image_id"]
        d = df.loc[ids].copy()
        d["avail"] = d["path"].isin(have)
        d["p_dr"] = sigmoid(P["dr"])
        d["p_gl"] = sigmoid(P["glaucoma"])
        d["p_cat"] = sigmoid(P["cataract"])
        d["grade_pred"] = P["dr_grade"].argmax(1)
        d = d[d.avail]
        # стадии ДР
        for g in range(5):
            c = d[(d.dr_grade == g) & (d.grade_pred == g)]
            if len(c):
                r = c.sort_values("p_dr", ascending=(g == 0)).iloc[0]
                out.append(dict(image_id=r.name, path=r.path, title=GRADE_RU[g], head="dr" if g else "dr",
                                prob=float(r.p_dr), split=split))
        for lab, col, head, ttl in (("glaucoma", "p_gl", "glaucoma", "Глаукома"),
                                    ("cataract", "p_cat", "cataract", "Катаракта")):
            c = d[d[lab] == 1].sort_values(col, ascending=False)
            for _, r in c.head(2).iterrows():
                out.append(dict(image_id=r.name, path=r.path, title=ttl, head=head, prob=float(r[col]), split=split))
    # убираем повторы по заголовку (берём по 1–2 на класс)
    seen, res = {}, []
    for e in out:
        k = e["title"]
        if seen.get(k, 0) < (2 if k in ("Глаукома", "Катаракта") else 1):
            res.append(e)
            seen[k] = seen.get(k, 0) + 1
    return res


# ============================================================================ Grad-CAM
def _forward_with_features(model, x):
    bb = model.backbone
    f = bb.forward_features(x)
    f.retain_grad()
    pooled = bb.forward_head(f, pre_logits=True)
    h = model.neck(pooled)
    out = {k: (v(h) if k == "dr_grade" else v(h).squeeze(1)) for k, v in model.heads.items()}
    return out, f


def _to_map(t, bb):
    if t.ndim == 3:  # ViT: (B, tokens, C) → (B, C, side, side)
        n_prefix = getattr(bb, "num_prefix_tokens", 1)
        tokens = t[:, n_prefix:]
        side = int(round(tokens.shape[1] ** 0.5))
        return tokens.transpose(1, 2).reshape(t.shape[0], -1, side, side)
    return t


def gradcam(model, img_rgb: np.ndarray, size: int, head="dr", device="cpu"):
    model.eval()
    im = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    x = torch.from_numpy(((im - MEAN) / STD).transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.enable_grad():
        out, f = _forward_with_features(model, x)
        target = out[head].sum()
        model.zero_grad(set_to_none=True)
        target.backward()
    fmap = _to_map(f.detach(), model.backbone)
    g = _to_map(f.grad, model.backbone)
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    cam -= cam.min()
    cam /= cam.max() + 1e-8
    # маска круга сетчатки — чтобы не светились чёрные поля
    gray = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    cam *= (gray > 12).astype(np.float32)
    return cam, float(sigmoid(out[head].detach().cpu().numpy())[0])


def overlay(img_rgb, cam, alpha=0.45):
    s = cam.shape[0]
    im = cv2.resize(img_rgb, (s, s))
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    m = np.clip((cam[..., None] - 0.1) / 0.9, 0, 1) * (alpha + 0.2)
    return (im * (1 - m) + heat * m).astype(np.uint8)


def gradcam_figure(models: dict, examples: list[dict], local_dir, out_png, device="cpu"):
    """Ряд на каждый пример: исходник + тепловая карта от каждой модели."""
    cols = 1 + len(models)
    rows = len(examples)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.3 * rows), facecolor=BG)
    axes = np.atleast_2d(axes)
    for i, ex in enumerate(examples):
        img = cv2.cvtColor(cv2.imread(str(Path(local_dir) / ex["path"])), cv2.COLOR_BGR2RGB)
        axes[i, 0].imshow(cv2.resize(img, (384, 384)))
        axes[i, 0].set_title(f"{ex['title']}", color=FG, fontsize=11)
        for j, (name, (model, size)) in enumerate(models.items(), start=1):
            cam, p = gradcam(model, img, size, head=ex["head"], device=device)
            axes[i, j].imshow(overlay(img, cam))
            axes[i, j].set_title(f"{name}: {100 * p:.0f}%", color=FG, fontsize=10)
        for a in axes[i]:
            a.axis("off")
    fig.suptitle("Куда смотрит ИИ: тепловые карты внимания (Grad-CAM)", color=FG, fontsize=15, y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=BG)
    plt.close(fig)
    return out_png


def _consensus_cam(models: dict, img, head, device, out_size=448):
    """Средняя карта внимания всех моделей («консилиум») + средняя вероятность."""
    cams, probs = [], []
    for model, size in models.values():
        cam, p = gradcam(model, img, size, head=head, device=device)
        cams.append(cv2.resize(cam, (out_size, out_size), interpolation=cv2.INTER_LINEAR))
        probs.append(p)
    cam = cv2.GaussianBlur(np.mean(cams, 0).astype(np.float32), (0, 0), out_size / 70)
    hi = np.percentile(cam[cam > 0], 99.5) if (cam > 0).any() else 1.0
    cam = np.clip(cam / (hi + 1e-8), 0, 1) ** 0.8
    return cam, float(np.mean(probs))


def glow_overlay(img_rgb, cam):
    """Для слайда: снимок слегка затемнён, внимание ИИ «светится» поверх."""
    s = cam.shape[0]
    im = cv2.resize(img_rgb, (s, s), interpolation=cv2.INTER_AREA).astype(np.float32)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1].astype(np.float32)
    m = np.clip((cam[..., None] - 0.15) / 0.6, 0, 1) * 0.85
    m *= (cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_RGB2GRAY) > 12)[..., None]
    base = im * 0.6
    return np.clip(base * (1 - m) + heat * m, 0, 255).astype(np.uint8)


def gradcam_slide(models: dict, examples: list[dict], local_dir, out_png, device="cpu", max_cols=6):
    """Слайд 16:9 — сверху снимок, снизу куда смотрит ИИ (консенсус всех моделей)."""
    order = ["нет ДР", "ДР 2 (умеренная)", "ДР 4 (пролиферативная)", "ДР 3 (тяжёлая)", "Глаукома", "Катаракта",
             "ДР 1 (лёгкая)"]
    chosen, used = [], set()
    for t in order:
        for e in examples:
            if e["title"] == t and t not in used:
                chosen.append(e)
                used.add(t)
                break
    chosen = sorted(chosen[:max_cols], key=lambda e: order.index(e["title"]))
    n = len(chosen)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7.6), facecolor=BG, gridspec_kw=dict(hspace=0.12, wspace=0.04))
    axes = np.array(axes).reshape(2, n)
    for j, ex in enumerate(chosen):
        img = cv2.cvtColor(cv2.imread(str(Path(local_dir) / ex["path"])), cv2.COLOR_BGR2RGB)
        cam, p = _consensus_cam(models, img, ex["head"], device)
        sick = ex["title"] != "нет ДР"
        verdict = f"{100 * p:.0f}%" if sick else f"здоров · {100 * (1 - p):.0f}%"
        axes[0, j].imshow(cv2.resize(img, (448, 448), interpolation=cv2.INTER_AREA))
        axes[0, j].set_title(ex["title"], color=FG, fontsize=13, pad=8)
        axes[1, j].imshow(glow_overlay(img, cam))
        axes[1, j].set_title(f"ИИ: {verdict}", color="#EF9F27" if sick else "#5DCAA5", fontsize=13, pad=8)
        for a in axes[:, j]:
            a.axis("off")
    k = len(models)
    who = "одной нейросети" if k == 1 else f"консилиума из {k} нейросетей"
    fig.suptitle(f"Куда смотрит ИИ Veya: тепловая карта внимания {who}", color=FG, fontsize=17, y=0.99)
    fig.text(0.5, 0.015, "Светлые пятна — участки, на которых нейросеть основывает диагноз "
             "(кровоизлияния, экссудаты, диск зрительного нерва, помутнение хрусталика).",
             ha="center", color="#9AA0A6", fontsize=11)
    fig.savefig(out_png, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ============================================================================ map of images
def embedding_map(pred_npz, labels: pd.DataFrame, out_png, max_points=15000, seed=0):
    """«Карта» снимков: внутреннее представление модели, сжатое до 2D (UMAP/t-SNE)."""
    P = np.load(pred_npz, allow_pickle=True)
    ids, emb = P["image_id"], P["emb"].astype(np.float32)
    if len(ids) > max_points:
        idx = np.random.default_rng(seed).choice(len(ids), max_points, replace=False)
        ids, emb = ids[idx], emb[idx]
    d = labels.set_index("image_id").loc[ids]
    from sklearn.decomposition import PCA
    z = PCA(min(50, len(emb) - 1, emb.shape[1]), random_state=seed).fit_transform(emb)
    try:
        import umap
        xy = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=seed).fit_transform(z)
        method = "UMAP"
    except Exception:
        from sklearn.manifold import TSNE
        xy = TSNE(2, perplexity=min(40, max(5, len(z) // 4)), init="pca", random_state=seed).fit_transform(z)
        method = "t-SNE"
    cat = np.full(len(d), "other", dtype=object)
    cat[(d.dr == 0).values & (d.glaucoma != 1).values & (d.cataract != 1).values] = "healthy"
    cat[(d.dr == 1).values] = "dr"
    cat[(d.glaucoma == 1).values] = "glaucoma"
    cat[(d.cataract == 1).values] = "cataract"
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
    ax.set_facecolor(BG)
    ax.scatter(xy[cat == "other", 0], xy[cat == "other", 1], s=2, c="#444441", alpha=0.3, linewidths=0)
    labels_ru = {"healthy": "Здоровые", "dr": "Диабетическая ретинопатия", "glaucoma": "Глаукома", "cataract": "Катаракта"}
    for k in ["healthy", "dr", "glaucoma", "cataract"]:
        m = cat == k
        ax.scatter(xy[m, 0], xy[m, 1], s=4 if k in ("healthy", "dr") else 9, c=ACCENT[k], alpha=0.7,
                   linewidths=0, label=f"{labels_ru[k]} ({m.sum()})")
    ax.legend(loc="lower left", frameon=False, labelcolor=FG, fontsize=12, markerscale=4)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Как ИИ «видит» {len(d):,} снимков глаза — карта ({method}).\n"
                 "Каждая точка — снимок; похожие для нейросети снимки оказываются рядом.",
                 color=FG, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, facecolor=BG)
    plt.close(fig)
    return out_png


# ============================================================================ ROC
def roc_figure(pred_npzs: dict, labels: pd.DataFrame, out_png, title):
    from sklearn.metrics import roc_auc_score, roc_curve
    df = labels.set_index("image_id")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG)
    for ax, k in zip(axes, BINARY):
        ax.set_facecolor(BG)
        for name, f in pred_npzs.items():
            if not Path(f).exists():
                continue
            P = np.load(f, allow_pickle=True)
            y = df.loc[P["image_id"], k].values
            m = y >= 0
            if m.sum() < 20 or y[m].min() == y[m].max():
                continue
            p = sigmoid(P[k][m])
            fpr, tpr, _ = roc_curve(y[m], p)
            ax.plot(fpr, tpr, lw=2, label=f"{name}: AUC {roc_auc_score(y[m], p):.3f}")
        ax.plot([0, 1], [0, 1], ls="--", c="#5F5E5A", lw=1)
        ax.set_title(NAMES_RU[k], color=FG, fontsize=13)
        ax.set_xlabel("Ложные тревоги (1 − специфичность)", color=FG)
        ax.set_ylabel("Найдено больных (чувствительность)", color=FG)
        ax.tick_params(colors=FG)
        for s in ax.spines.values():
            s.set_color("#5F5E5A")
        ax.legend(frameon=False, labelcolor=FG, loc="lower right")
    fig.suptitle(title, color=FG, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, facecolor=BG)
    plt.close(fig)
    return out_png
