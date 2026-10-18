"""VEYA v2 — обучение базовых моделей ансамбля (multi-task: стадия ДР, ДР, глаукома, катаракта).

Модели («глаза» ансамбля):
  effv2s   — EfficientNetV2-S, 512px  (CNN, ловит мелкие поражения: микроаневризмы, экссудаты)
  convnext — ConvNeXt-Small, 384px    (современная CNN, другие ошибки чем у EfficientNet)
  vit      — ViT-B/14 DINOv2, 224px   (трансформер, смотрит на снимок целиком)
  retfound — RETFound ViT-L, 224px    (foundation-модель, предобучена на 1.6 млн снимков сетчатки;
                                       нужен HF_TOKEN в секретах Colab, иначе используйте vit)

Всё устойчиво к обрывам Colab: чекпоинт на Google Drive каждые N минут, запуск заново — продолжает с места.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x

BINARY = ["dr", "glaucoma", "cataract"]
HEADS = ["dr_grade"] + BINARY

MODELS = {
    "effv2s": dict(timm="tf_efficientnetv2_s.in21k_ft_in1k", size=512, bs=16, accum=2, lr=3e-4, epochs=14,
                   samples_per_epoch=36000, channels_last=True),
    "convnext": dict(timm="convnext_small.fb_in22k_ft_in1k_384", size=384, bs=16, accum=2, lr=1.5e-4, epochs=12,
                     samples_per_epoch=36000, channels_last=True, drop_path=0.2),
    "vit": dict(timm="vit_base_patch14_dinov2.lvd142m", size=224, bs=32, accum=1, lr=4e-5, epochs=12,
                samples_per_epoch=36000, kwargs=dict(img_size=224), drop_path=0.1, layer_decay=0.75),
    "retfound": dict(timm="vit_large_patch16_224", size=224, bs=16, accum=2, lr=3e-5, epochs=10,
                     samples_per_epoch=36000, kwargs=dict(global_pool="avg"), drop_path=0.2, layer_decay=0.65,
                     hf=("YukunZhou/RETFound_mae_natureCFP", "RETFound_mae_natureCFP.pth")),
}

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


# ============================================================================ data
def load_image(path: str, size: int) -> np.ndarray:
    flag = cv2.IMREAD_REDUCED_COLOR_2 if size <= 256 else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        img = np.zeros((size, size, 3), np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Аугментации, безопасные для глазного дна (не ломают признаки болезней)."""
    s = img.shape[0]
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.3:
        img = img[::-1]
    ang = rng.uniform(-180, 180)
    scale = rng.uniform(0.92, 1.12)
    tx, ty = rng.uniform(-0.04, 0.04, 2) * s
    M = cv2.getRotationMatrix2D((s / 2, s / 2), ang, scale)
    M[:, 2] += (tx, ty)
    img = cv2.warpAffine(np.ascontiguousarray(img), M, (s, s), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    img = img.astype(np.float32)
    # яркость / контраст / гамма / баланс каналов (разные камеры и смартфон-насадка)
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    mean = img.mean()
    img = (img - mean) * rng.uniform(0.8, 1.25) + mean
    img *= rng.uniform(0.92, 1.08, size=3)[None, None, :]
    img = np.clip(img, 0, 255)
    if rng.random() < 0.3:
        g = rng.uniform(0.8, 1.25)
        img = 255.0 * (img / 255.0) ** g
    if rng.random() < 0.2:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.15:
        img = img + rng.normal(0, rng.uniform(2, 8), img.shape)
    return np.clip(img, 0, 255)


class FundusDS(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, size: int, train: bool):
        self.paths = [os.path.join(root, p) for p in df["path"].values]
        self.y = df[HEADS].values.astype(np.int64)
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_image(self.paths[i], self.size)
        if self.train:
            rng = np.random.default_rng((os.getpid() * 1_000_003 + i * 7919 + int(time.time() * 1e6)) % 2**32)
            img = augment(img, rng)
        else:
            img = img.astype(np.float32)
        img = ((img - MEAN) / STD).astype(np.float32)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        return x, torch.from_numpy(self.y[i])


class EpochSampler(Sampler):
    """Взвешенная выборка фиксированного размера на эпоху (баланс датасетов и редких болезней).
    Детерминирована по номеру эпохи → после обрыва Colab продолжаем с того же шага."""

    def __init__(self, weights: np.ndarray, n: int, epoch: int, start: int = 0, seed: int = 2026):
        rng = np.random.default_rng(seed + epoch)
        p = weights / weights.sum()
        self.idx = rng.choice(len(weights), size=n, replace=True, p=p)[start:]

    def __iter__(self):
        return iter(self.idx.tolist())

    def __len__(self):
        return len(self.idx)


def sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Вес снимка ~ 1/sqrt(размер его «страты»): страта = датасет + комбинация меток.
    Больные глаукомой/катарактой/тяжёлой ДР попадают в батчи заметно чаще."""
    key = (df["dataset"].astype(str) + "|" + df["dr_grade"].astype(str) + "|" +
           df["glaucoma"].astype(str) + "|" + df["cataract"].astype(str))
    cnt = key.map(key.value_counts())
    w = 1.0 / np.sqrt(cnt.values.astype(np.float64))
    # дополнительно поднимаем редкие позитивы
    w *= np.where(df["glaucoma"].values == 1, 1.5, 1.0)
    w *= np.where(df["cataract"].values == 1, 2.0, 1.0)
    w *= np.where(df["dr_grade"].values >= 3, 1.5, 1.0)
    return w


# ============================================================================ model
class VeyaNet(nn.Module):
    """Backbone + общий многослойный «нейрон-мозг» модели + 4 головы."""

    def __init__(self, name: str, pretrained=True, drop=0.3, drop_path=0.0, **kw):
        super().__init__()
        import timm
        extra = dict(kw)
        if drop_path:
            extra["drop_path_rate"] = drop_path
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, **extra)
        d = self.backbone.num_features
        self.neck = nn.Sequential(
            nn.Dropout(drop), nn.Linear(d, 768), nn.GELU(),
            nn.Dropout(drop / 2), nn.Linear(768, 384), nn.GELU(),
        )
        self.heads = nn.ModuleDict({
            "dr_grade": nn.Linear(384, 5),
            **{k: nn.Linear(384, 1) for k in BINARY},
        })
        self.emb_dim = d

    def forward(self, x, return_emb=False):
        f = self.backbone(x)
        h = self.neck(f)
        out = {k: (v(h) if k == "dr_grade" else v(h).squeeze(1)) for k, v in self.heads.items()}
        return (out, f) if return_emb else out


def load_retfound(model: VeyaNet, repo: str, fname: str):
    from huggingface_hub import hf_hub_download
    token = os.environ.get("HF_TOKEN")
    path = hf_hub_download(repo, fname, token=token)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck.get("teacher", ck))
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith(("decoder", "mask_token", "head"))}
    msg = model.backbone.load_state_dict(sd, strict=False)
    print(f"RETFound загружен. missing={len(msg.missing_keys)} {msg.missing_keys[:6]} "
          f"unexpected={len(msg.unexpected_keys)} {msg.unexpected_keys[:6]}")


def build_model(key: str, pretrained=True) -> VeyaNet:
    cfg = MODELS[key]
    m = VeyaNet(cfg["timm"], pretrained=pretrained and "hf" not in cfg,
                drop_path=cfg.get("drop_path", 0.0), **cfg.get("kwargs", {}))
    if pretrained and "hf" in cfg:
        load_retfound(m, *cfg["hf"])
    return m


def param_groups(model: VeyaNet, lr: float, wd=0.05, layer_decay: float | None = None):
    """Голова учится в 10 раз быстрее; для ViT — послойное затухание lr (нижние слои меняются меньше)."""
    head = list(model.neck.parameters()) + list(model.heads.parameters())
    groups = [dict(params=head, lr=lr * 10, weight_decay=wd)]
    bb = model.backbone
    if layer_decay and hasattr(bb, "blocks"):
        n = len(bb.blocks)
        layers = {}
        for name, p in bb.named_parameters():
            if name.startswith("blocks."):
                lid = int(name.split(".")[1]) + 1
            elif name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
                lid = 0
            else:
                lid = n + 1
            no_wd = p.ndim == 1 or name.endswith(".bias") or "token" in name or "pos_embed" in name
            layers.setdefault((lid, no_wd), []).append(p)
        for (lid, no_wd), ps in layers.items():
            groups.append(dict(params=ps, lr=lr * layer_decay ** (n + 1 - lid), weight_decay=0.0 if no_wd else wd))
    else:
        dec, nodec = [], []
        for name, p in bb.named_parameters():
            (nodec if p.ndim == 1 or name.endswith(".bias") else dec).append(p)
        groups += [dict(params=dec, lr=lr, weight_decay=wd), dict(params=nodec, lr=lr, weight_decay=0.0)]
    for g in groups:
        g["base_lr"] = g["lr"]
    return groups


# ============================================================================ loss + metrics
def multitask_loss(out, y, pos_weight: dict, smoothing=0.05):
    losses = {}
    yg = y[:, 0]
    m = yg >= 0
    if m.any():
        losses["dr_grade"] = F.cross_entropy(out["dr_grade"][m].float(), yg[m], label_smoothing=smoothing)
    for j, k in enumerate(BINARY, start=1):
        yy = y[:, j]
        m = yy >= 0
        if m.any():
            t = yy[m].float() * (1 - smoothing) + smoothing / 2
            losses[k] = F.binary_cross_entropy_with_logits(
                out[k][m].float(), t, pos_weight=torch.tensor(pos_weight[k], device=t.device))
    total = sum(losses.values()) if losses else out["dr"].sum() * 0
    return total, {k: float(v.detach()) for k, v in losses.items()}


def compute_metrics(logits: dict, y: np.ndarray) -> dict:
    res = {}
    yg = y[:, 0]
    m = yg >= 0
    if m.sum() > 10:
        pred = logits["dr_grade"][m].argmax(1)
        res["dr_grade_acc"] = accuracy_score(yg[m], pred)
        res["dr_grade_qwk"] = cohen_kappa_score(yg[m], pred, weights="quadratic")
    for j, k in enumerate(BINARY, start=1):
        yy = y[:, j]
        m = yy >= 0
        if m.sum() > 10 and 0 < yy[m].sum() < m.sum():
            p = 1 / (1 + np.exp(-logits[k][m]))
            res[f"{k}_auc"] = roc_auc_score(yy[m], p)
            res[f"{k}_bal_acc"] = 0.5 * ((p[yy[m] == 1] >= 0.5).mean() + (p[yy[m] == 0] < 0.5).mean())
    keys = [res.get("dr_auc"), res.get("glaucoma_auc"), res.get("cataract_auc"), res.get("dr_grade_qwk")]
    keys = [k for k in keys if k is not None]
    res["score"] = float(np.mean(keys)) if keys else 0.0
    return res


# ============================================================================ checkpoint helpers
def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_local = Path("/tmp") / (path.name + ".tmp")
    torch.save(obj, tmp_local)
    tmp_remote = path.with_suffix(path.suffix + ".tmp")
    shutil.copy(tmp_local, tmp_remote)
    os.replace(tmp_remote, path)
    tmp_local.unlink(missing_ok=True)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def copy_to(self, model):
        model.load_state_dict({k: v.to(dtype=model.state_dict()[k].dtype) for k, v in self.shadow.items()})


# ============================================================================ train / eval
@torch.no_grad()
def predict(model, df, root, size, bs=32, tta=True, device="cuda", return_emb=False, workers=2):
    model.eval()
    dl = DataLoader(FundusDS(df, root, size, train=False), batch_size=bs, shuffle=False,
                    num_workers=workers, pin_memory=True)
    outs = {k: [] for k in HEADS}
    embs = []
    for x, _ in tqdm(dl, desc="predict", leave=False):
        x = x.to(device, non_blocking=True)
        views = [x, torch.flip(x, dims=[3])] if tta else [x]
        acc, eacc = None, None
        for v in views:
            with torch.autocast(device_type="cuda" if device != "cpu" else "cpu", dtype=torch.float16,
                                enabled=device != "cpu"):
                o, e = model(v, return_emb=True)
            o = {k: t.float() for k, t in o.items()}
            acc = o if acc is None else {k: acc[k] + o[k] for k in o}
            eacc = e.float() if eacc is None else eacc + e.float()
        for k in HEADS:
            outs[k].append((acc[k] / len(views)).cpu().numpy())
        if return_emb:
            embs.append((eacc / len(views)).cpu().numpy().astype(np.float16))
    res = {k: np.concatenate(v) for k, v in outs.items()}
    if return_emb:
        res["emb"] = np.concatenate(embs)
    return res


def train_model(key: str, data_dir="/content/veya_data", drive_dir="/content/drive/MyDrive/veya_v2",
                epochs: int | None = None, ckpt_minutes=12, workers=2, max_hours: float | None = None,
                quick_test=False, pretrained=True):
    cfg = dict(MODELS[key])
    if epochs:
        cfg["epochs"] = epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(drive_dir) / "models" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(Path(data_dir) / "labels.csv")
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    if quick_test:
        tr, va = tr.head(64), va.head(32)
        cfg.update(samples_per_epoch=32, bs=8, accum=1)
    print(f"[{key}] train={len(tr)} val={len(va)} size={cfg['size']} device={device}")

    pos_weight = {}
    for k in BINARY:
        y = tr[k][tr[k] >= 0]
        pos = max(1, int(y.sum()))
        pos_weight[k] = float(min(5.0, math.sqrt((len(y) - pos) / pos)))
    print("pos_weight:", pos_weight)

    model = build_model(key, pretrained=pretrained and not quick_test).to(device)
    if cfg.get("channels_last") and device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model, cfg["lr"], layer_decay=cfg.get("layer_decay")), betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ema = EMA(model, decay=0.999 if not quick_test else 0.5)
    steps_per_epoch = cfg["samples_per_epoch"] // (cfg["bs"] * cfg["accum"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = max(1, int(0.03 * total_steps))

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t))

    state = dict(epoch=0, step_in_epoch=0, global_step=0, best=-1.0, history=[])
    last = out_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        ema.shadow = {k: v.float().to(device) for k, v in ck["ema"].items()}
        state = ck["state"]
        print(f"▶ Продолжаю с эпохи {state['epoch']+1}, шаг {state['step_in_epoch']} (best={state['best']:.4f})")
    if state.get("done"):
        print("Модель уже обучена. Удалите last.pt чтобы обучить заново.")
        return out_dir

    weights = sample_weights(tr)
    ds = FundusDS(tr, data_dir, cfg["size"], train=True)
    t_start = time.time()
    last_ck = time.time()

    def save_last():
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(),
                         ema=ema.shadow, state=state, cfg=cfg), last)

    while state["epoch"] < cfg["epochs"]:
        ep = state["epoch"]
        sampler = EpochSampler(weights, steps_per_epoch * cfg["bs"] * cfg["accum"], ep,
                               start=state["step_in_epoch"] * cfg["bs"] * cfg["accum"])
        dl = DataLoader(ds, batch_size=cfg["bs"], sampler=sampler, num_workers=workers, pin_memory=True,
                        drop_last=True, persistent_workers=False, prefetch_factor=4 if workers else None)
        model.train()
        pbar = tqdm(dl, desc=f"[{key}] epoch {ep+1}/{cfg['epochs']}", initial=state["step_in_epoch"] * cfg["accum"],
                    total=steps_per_epoch * cfg["accum"])
        run = {}
        for it, (x, y) in enumerate(pbar):
            x = x.to(device, non_blocking=True)
            if cfg.get("channels_last") and device == "cuda":
                x = x.to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=torch.float16,
                                enabled=device == "cuda"):
                out = model(x)
            loss, parts = multitask_loss(out, y, pos_weight)
            scaler.scale(loss / cfg["accum"]).backward()
            for k, v in parts.items():
                run[k] = 0.98 * run.get(k, v) + 0.02 * v
            if (it + 1) % cfg["accum"] == 0:
                f = lr_factor(state["global_step"])
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model)
                state["global_step"] += 1
                state["step_in_epoch"] += 1
                pbar.set_postfix({k: f"{v:.3f}" for k, v in run.items()})
                if time.time() - last_ck > ckpt_minutes * 60:
                    save_last()
                    last_ck = time.time()
                if max_hours and time.time() - t_start > max_hours * 3600:
                    save_last()
                    print(f"⏸ Лимит времени {max_hours}ч — чекпоинт сохранён, перезапустите ячейку позже.")
                    return out_dir
        # ---- конец эпохи: валидация EMA-весов
        live = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)
        logits = predict(model, va, data_dir, cfg["size"], bs=cfg["bs"] * 2, tta=False, device=device, workers=workers)
        met = compute_metrics(logits, va[HEADS].values)
        met["epoch"] = ep + 1
        met["train_loss"] = {k: round(v, 4) for k, v in run.items()}
        state["history"].append(met)
        print(f"[{key}] epoch {ep+1}: " + ", ".join(f"{k}={v:.4f}" for k, v in met.items() if isinstance(v, float)))
        if met["score"] > state["best"]:
            state["best"] = met["score"]
            atomic_save(dict(model=model.state_dict(), cfg=cfg, metrics=met, key=key), out_dir / "best.pt")
            print(f"  ★ новый лучший score={met['score']:.4f} → best.pt")
        model.load_state_dict(live)
        state["epoch"] += 1
        state["step_in_epoch"] = 0
        save_last()
        last_ck = time.time()
        (out_dir / "history.json").write_text(json.dumps(state["history"], indent=1, default=float))
    state["done"] = True
    save_last()
    print(f"[{key}] обучение завершено. best score={state['best']:.4f}")
    return out_dir


def export_predictions(key: str, data_dir="/content/veya_data", drive_dir="/content/drive/MyDrive/veya_v2",
                       splits=("val", "test", "test_ext"), workers=2):
    """Предсказания лучшей модели (с TTA) + эмбеддинги → вход для MLP-«мозга»."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(drive_dir) / "models" / key
    ck = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(key, pretrained=False)
    model.load_state_dict(ck["model"])
    model.to(device)
    df = pd.read_csv(Path(data_dir) / "labels.csv")
    pred_dir = Path(drive_dir) / "preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for sp in splits:
        d = df[df.split == sp].reset_index(drop=True)
        if not len(d):
            continue
        r = predict(model, d, data_dir, ck["cfg"]["size"], bs=ck["cfg"]["bs"] * 2, tta=True,
                    device=device, return_emb=True, workers=workers)
        np.savez_compressed(pred_dir / f"{key}_{sp}.npz", image_id=d["image_id"].values, **r)
        met = compute_metrics(r, d[HEADS].values)
        print(f"[{key}] {sp}: " + ", ".join(f"{k}={v:.4f}" for k, v in met.items()))
    return pred_dir
