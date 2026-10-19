"""VEYA v2 — «Мозг»: многослойная нейросеть, которая принимает решения по ответам всех моделей.

Каждая базовая модель («глаз») выдаёт:  вероятности по 4 задачам + эмбеддинг снимка.
Мозг (MLP 3 слоя) получает всё это вместе и выдаёт итоговые вероятности.
Обучается на val-части (её базовые модели не видели), проверяется на test и test_ext.

Плюс:
  * пороги по каждой болезни подбираются на val (а не тупо 0.5)
  * «зона сомнения»: если мозг не уверен — ответ «нужна проверка врача».
    Так модель смело говорит там, где уверена, и не врёт там, где не уверена.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import GroupKFold

BINARY = ["dr", "glaucoma", "cataract"]
HEADS = ["dr_grade"] + BINARY
NAMES_RU = {"dr": "Диабетическая ретинопатия", "glaucoma": "Глаукома", "cataract": "Катаракта"}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ============================================================================ features
def load_preds(pred_dir, models, split):
    out = {}
    for m in models:
        f = Path(pred_dir) / f"{m}_{split}.npz"
        if f.exists():
            out[m] = dict(np.load(f, allow_pickle=True))
    return align_preds(out)


def align_preds(out):
    """Модели с Kaggle обучены на чуть другом делении (±20 снимков). Оставляем только снимки,
    которые попали в эту же часть у ВСЕХ моделей, и ставим их в одинаковом порядке —
    так ни одна модель не оценивается на снимке, который видела при обучении."""
    if len(out) < 2:
        return out
    ids0 = [str(i) for i in next(iter(out.values()))["image_id"]]
    common = set(ids0)
    for p in out.values():
        common &= set(str(i) for i in p["image_id"])
    keep = [i for i in ids0 if i in common]
    for m, p in out.items():
        ids = [str(i) for i in p["image_id"]]
        n = len(ids)
        pos = {i: k for k, i in enumerate(ids)}
        idx = np.array([pos[i] for i in keep], dtype=np.int64)
        out[m] = {k: (v[idx] if hasattr(v, "shape") and v.ndim >= 1 and len(v) == n else v) for k, v in p.items()}
        if len(keep) != n:
            print(f"  [{m}] выравнивание: {n} → {len(keep)} общих снимков")
    return out


class FeatureBuilder:
    """Признаки для мозга: вероятности каждой модели + PCA-сжатые эмбеддинги."""

    def __init__(self, models, n_pca=16):
        self.models = models
        self.n_pca = n_pca
        self.pca = {}
        self.mu = None
        self.sd = None

    def _raw(self, preds, fit=False):
        cols = []
        for m in self.models:
            p = preds[m]
            cols.append(softmax(p["dr_grade"].astype(np.float32)))
            cols.append(np.stack([p[k] for k in BINARY], 1).astype(np.float32))  # логиты
            e = p["emb"].astype(np.float32)
            if fit:
                self.pca[m] = PCA(min(self.n_pca, len(e) - 1, e.shape[1]), random_state=0).fit(e)
            cols.append(self.pca[m].transform(e).astype(np.float32))
        return np.concatenate(cols, 1)

    def fit_transform(self, preds):
        x = self._raw(preds, fit=True)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        return (x - self.mu) / self.sd, self.base(preds)

    def transform(self, preds):
        return (self._raw(preds) - self.mu) / self.sd, self.base(preds)

    def base(self, preds):
        """Среднее мнение моделей (логиты) — мозг учится его ПОПРАВЛЯТЬ, а не угадывать с нуля."""
        g = np.mean([preds[m]["dr_grade"].astype(np.float32) for m in self.models], 0)
        b = np.mean([np.stack([preds[m][k] for k in BINARY], 1).astype(np.float32) for m in self.models], 0)
        return np.concatenate([g, b], 1)


# ============================================================================ brain network
class Brain(nn.Module):
    def __init__(self, d_in, hidden=(256, 128, 64), drop=0.3):
        super().__init__()
        layers, d = [], d_in
        for i, h in enumerate(hidden):
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(drop if i == 0 else drop / 2)]
            d = h
        self.body = nn.Sequential(*layers)
        self.grade = nn.Linear(d, 5)
        self.bin = nn.Linear(d, 3)
        for lin in (self.grade, self.bin):  # старт = ровно среднее мнение моделей
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x, base):
        h = self.body(x)
        return self.grade(h) + base[:, :5], self.bin(h) + base[:, 5:]


def brain_loss(grade, binl, y):
    loss = 0.0
    m = y[:, 0] >= 0
    if m.any():
        loss = loss + F.cross_entropy(grade[m], y[m, 0])
    for j in range(3):
        mm = y[:, j + 1] >= 0
        if mm.any():
            loss = loss + F.binary_cross_entropy_with_logits(binl[mm, j], y[mm, j + 1].float())
    return loss


def fit_brain(x, base, y, epochs=40, lr=1e-3, wd=1e-2, seed=0, device="cpu"):
    torch.manual_seed(seed)
    net = Brain(x.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(x) // 256 + 1))
    X = torch.tensor(x, dtype=torch.float32, device=device)
    B0 = torch.tensor(base, dtype=torch.float32, device=device)
    Y = torch.tensor(y, dtype=torch.long, device=device)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]
            if len(b) < 2:
                continue
            xb = X[b] + 0.05 * torch.randn_like(X[b])  # шум = регуляризация
            gl, bl = net(xb, B0[b])
            loss = brain_loss(gl, bl, Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    return net


@torch.no_grad()
def brain_predict(nets, x, base, device="cpu"):
    X = torch.tensor(x, dtype=torch.float32, device=device)
    B0 = torch.tensor(base, dtype=torch.float32, device=device)
    gs, bs = [], []
    for n in nets:
        g, b = n(X, B0)
        gs.append(F.softmax(g, 1).cpu().numpy())
        bs.append(torch.sigmoid(b).cpu().numpy())
    return np.mean(gs, 0), np.mean(bs, 0)


# ============================================================================ thresholds + doubt zone
def best_threshold(y, p):
    """Порог, максимизирующий сбалансированную точность (чувствительность+специфичность)/2."""
    ts = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    best, bt = -1, 0.5
    for t in ts:
        ba = 0.5 * ((p[y == 1] >= t).mean() + (p[y == 0] < t).mean())
        if ba > best:
            best, bt = ba, t
    return float(bt)


def doubt_zone(y, p, thr, target=0.95, max_refer=0.25):
    """Ширина «зоны сомнения» вокруг порога: самая узкая, при которой точность на уверенных ≥ target."""
    for w in np.linspace(0, 0.45, 91):
        lo, hi = max(0.0, thr - w), min(1.0, thr + w)
        conf = (p < lo) | (p >= hi)
        if conf.mean() < 1 - max_refer:
            break
        if conf.sum() and ((p[conf] >= thr) == y[conf]).mean() >= target:
            return float(lo), float(hi)
    w = 0.0
    for w2 in np.linspace(0, 0.45, 91):
        lo, hi = max(0.0, thr - w2), min(1.0, thr + w2)
        if ((p < lo) | (p >= hi)).mean() < 1 - max_refer:
            break
        w = w2
    return float(max(0.0, thr - w)), float(min(1.0, thr + w))


def disease_report(y, p, thr, zone):
    m = y >= 0
    y, p = y[m].astype(int), p[m]
    if len(y) == 0 or y.min() == y.max():
        return None
    pred = (p >= thr).astype(int)
    conf = (p < zone[0]) | (p >= zone[1])
    tp = ((pred == 1) & (y == 1)).sum()
    tn = ((pred == 0) & (y == 0)).sum()
    return dict(
        n=int(len(y)), positives=int(y.sum()),
        auc=float(roc_auc_score(y, p)),
        accuracy=float((pred == y).mean()),
        sensitivity=float(tp / max(1, y.sum())),
        specificity=float(tn / max(1, (y == 0).sum())),
        balanced_accuracy=float(0.5 * (tp / max(1, y.sum()) + tn / max(1, (y == 0).sum()))),
        confident_share=float(conf.mean()),
        accuracy_confident=float((pred[conf] == y[conf]).mean()) if conf.any() else float("nan"),
    )


# ============================================================================ main
def run_brain(models=("effv2s", "convnext", "vit"), data_dir="/content/veya_data",
              drive_dir="/content/drive/MyDrive/veya_v2", folds=5, seeds=3, device=None, always_brain=True):
    """always_brain=True: the brain makes every final decision (thresholds are tuned on its own outputs).
    False: per disease, fall back to the plain average where the brain is not better on val."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pred_dir = Path(drive_dir) / "preds"
    labels = pd.read_csv(Path(data_dir) / "labels.csv").set_index("image_id")
    val = load_preds(pred_dir, models, "val")
    models = [m for m in models if m in val]
    print("Модели в ансамбле:", models)
    assert models, "Нет предсказаний. Сначала обучите модели и запустите export_predictions."
    ids = val[models[0]]["image_id"]
    for m in models:
        assert (val[m]["image_id"] == ids).all(), f"порядок снимков не совпадает у {m}"
    yv = labels.loc[ids, HEADS].values.astype(np.int64)
    groups = labels.loc[ids, "group"].values

    fb = FeatureBuilder(models)
    xv, bv = fb.fit_transform(val)
    print(f"Признаков для мозга: {xv.shape[1]}, обучающих снимков (val): {len(xv)}")

    # --- кросс-валидация на val: честные OOF-вероятности для подбора порогов
    oof_g = np.zeros((len(xv), 5))
    oof_b = np.zeros((len(xv), 3))
    for f, (tr, te) in enumerate(GroupKFold(folds).split(xv, groups=groups)):
        nets = [fit_brain(xv[tr], bv[tr], yv[tr], seed=s, device=device) for s in range(seeds)]
        g, b = brain_predict(nets, xv[te], bv[te], device)
        oof_g[te], oof_b[te] = g, b
    # сравнение с простым усреднением моделей
    avg_b = np.mean([sigmoid(np.stack([val[m][k] for k in BINARY], 1)) for m in models], 0)

    thresholds, zones, cmp, use_brain = {}, {}, {}, {}
    for j, k in enumerate(BINARY):
        m = yv[:, j + 1] >= 0
        if m.sum() == 0 or yv[m, j + 1].min() == yv[m, j + 1].max():
            continue
        cmp[k] = dict(brain_auc=roc_auc_score(yv[m, j + 1], oof_b[m, j]),
                      average_auc=roc_auc_score(yv[m, j + 1], avg_b[m, j]))
        # по умолчанию решает мозг; с always_brain=False — только там, где он лучше усреднения на val
        use_brain[k] = True if always_brain else bool(cmp[k]["brain_auc"] > cmp[k]["average_auc"])
        pv = oof_b[m, j] if use_brain[k] else avg_b[m, j]
        thresholds[k] = best_threshold(yv[m, j + 1], pv)
        zones[k] = doubt_zone(yv[m, j + 1], pv, thresholds[k])
    print("\nМозг vs простое усреднение (AUC на val, out-of-fold):")
    for k, v in cmp.items():
        print(f"  {NAMES_RU[k]:28s} мозг={v['brain_auc']:.4f}   среднее={v['average_auc']:.4f}   "
              f"→ {'мозг' if use_brain[k] else 'среднее'}")

    # --- финальный мозг на всём val
    final = [fit_brain(xv, bv, yv, seed=s, device=device) for s in range(seeds * 2)]

    report = {"models": list(models), "thresholds": thresholds, "doubt_zones": zones, "brain_vs_average": cmp,
              "use_brain": use_brain}
    for sp in ["test", "test_ext"]:
        P = load_preds(pred_dir, models, sp)
        if any(m not in P for m in models):
            continue
        ids_s = P[models[0]]["image_id"]
        ys = labels.loc[ids_s, HEADS].values.astype(np.int64)
        xs, bs_ = fb.transform(P)
        g, b = brain_predict(final, xs, bs_, device)
        avg_s = np.mean([sigmoid(np.stack([P[m][k] for k in BINARY], 1)) for m in models], 0)
        for j, k in enumerate(BINARY):
            if k in use_brain and not use_brain[k]:
                b[:, j] = avg_s[:, j]
        rep = {}
        for j, k in enumerate(BINARY):
            if k in thresholds:
                r = disease_report(ys[:, j + 1], b[:, j], thresholds[k], zones[k])
                if r:
                    rep[k] = r
        mg = ys[:, 0] >= 0
        if mg.sum() > 10:
            pg = g[mg].argmax(1)
            rep["dr_grade"] = dict(n=int(mg.sum()), accuracy=float(accuracy_score(ys[mg, 0], pg)),
                                   qwk=float(cohen_kappa_score(ys[mg, 0], pg, weights="quadratic")))
        # общая точность: доля правильных бинарных решений по всем размеченным болезням
        corr, tot, corr_c, tot_c = 0, 0, 0, 0
        for j, k in enumerate(BINARY):
            if k not in thresholds:
                continue
            mm = ys[:, j + 1] >= 0
            pr = b[mm, j] >= thresholds[k]
            corr += (pr == ys[mm, j + 1]).sum()
            tot += mm.sum()
            cf = (b[mm, j] < zones[k][0]) | (b[mm, j] >= zones[k][1])
            corr_c += (pr[cf] == ys[mm, j + 1][cf]).sum()
            tot_c += cf.sum()
        rep["overall"] = dict(accuracy=float(corr / max(1, tot)), decisions=int(tot),
                              accuracy_confident=float(corr_c / max(1, tot_c)),
                              confident_share=float(tot_c / max(1, tot)))
        # разбивка по источникам (видно, где модель переносится хуже)
        subs = labels.loc[ids_s, "subset"].values
        by_src = {}
        for s in np.unique(subs):
            ms = subs == s
            for j, k in enumerate(BINARY):
                if k not in thresholds:
                    continue
                r = disease_report(ys[ms, j + 1], b[ms, j], thresholds[k], zones[k])
                if r and r["n"] >= 30:
                    by_src[f"{s}/{k}"] = r
        rep["by_source"] = by_src
        report[sp] = rep
        print_report(sp, rep)
        if by_src:
            print("  по источникам:")
            for key, r in by_src.items():
                print(f"    {key:32s} n={r['n']:5d} AUC={r['auc']:.3f} bal_acc={r['balanced_accuracy']:.3f} "
                      f"sens={r['sensitivity']:.3f} spec={r['specificity']:.3f}")

    out = Path(drive_dir) / "brain"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(nets=[n.state_dict() for n in final], d_in=xv.shape[1], models=list(models),
                    pca={m: (fb.pca[m].components_, fb.pca[m].mean_) for m in models},
                    mu=fb.mu, sd=fb.sd, thresholds=thresholds, zones=zones, use_brain=use_brain), out / "brain.pt")
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))
    print(f"\nМозг и отчёт сохранены в {out}")
    return report


def print_report(split, rep):
    title = {"test": "ВНУТРЕННИЙ ТЕСТ (новые пациенты)", "test_ext": "ВНЕШНИЙ ТЕСТ (датасеты, которых модель не видела)"}
    print(f"\n===== {title.get(split, split)} =====")
    print(f"{'Болезнь':28s} {'n':>6s} {'AUC':>6s} {'Точн.':>6s} {'Чувств.':>7s} {'Спец.':>6s} {'Увер.%':>7s} {'Точн.увер.':>10s}")
    for k in BINARY:
        r = rep.get(k)
        if not r:
            continue
        print(f"{NAMES_RU[k]:28s} {r['n']:6d} {r['auc']:6.3f} {r['accuracy']:6.3f} {r['sensitivity']:7.3f} "
              f"{r['specificity']:6.3f} {100*r['confident_share']:6.1f}% {r['accuracy_confident']:10.3f}")
    if "dr_grade" in rep:
        r = rep["dr_grade"]
        print(f"Стадия ДР (0–4): accuracy={r['accuracy']:.3f}, QWK={r['qwk']:.3f} (n={r['n']})")
    o = rep["overall"]
    print(f"ОБЩАЯ ТОЧНОСТЬ: {100*o['accuracy']:.1f}%   |   на уверенных ответах: {100*o['accuracy_confident']:.1f}% "
          f"(уверенных {100*o['confident_share']:.0f}%, остальные → «нужен врач»)")
