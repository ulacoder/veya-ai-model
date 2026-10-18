"""VEYA v2 — сборка единого датасета глазного дна.

Что делает:
  1. Скачивает публичные датасеты (Kaggle через kagglehub, без токена).
  2. Читает метки и приводит их к одной схеме на КАЖДЫЙ ГЛАЗ:
        dr_grade  0..4  (стадия ДР, -1 = неизвестно)
        dr        0/1   (есть ли ДР, -1 = неизвестно)
        glaucoma  0/1   (-1 = неизвестно / подозрение)
        cataract  0/1   (-1 = неизвестно)
  3. Обрезает чёрные поля, приводит к 512x512, считает перцептивный хэш.
  4. Находит дубликаты между датасетами и сливает их метки.
  5. Делит по ПАЦИЕНТАМ (а не по снимкам): train / val / test,
     плюс test_ext — целые датасеты, которые модель вообще не видит (IDRiD, PAPILA, RIM-ONE).
  6. Сохраняет на Google Drive по частям (parts/*.tar) + labels.csv.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **k):
        return x

IMG_SIZE = 512
LABEL_COLS = ["dr_grade", "dr", "glaucoma", "cataract"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Kaggle-источники (все публичные, скачиваются без токена)
SOURCES = {
    "eyepacs_aptos": "benjaminwarner/resized-2015-2019-blindness-detection-images",  # EyePACS 2015 (train+test с метками) + APTOS 2019
    "ddr": "samriddhibagchi/ddr-dataset-credits-to-authors",
    "idrid": "aaryapatel98/indian-diabetic-retinopathy-image-dataset",
    "odir": "andrewmvd/ocular-disease-recognition-odir5k",
    "smdg": "deathtrooper/multichannel-glaucoma-benchmark-dataset",  # REFUGE, RIM-ONE, PAPILA, ORIGA, G1020, ACRIMA ...
    "airogs": "deathtrooper/glaucoma-dataset-eyepacs-airogs-light-v2",
}

# Датасеты, которые целиком уходят во внешний тест (модель их не видит при обучении)
EXTERNAL_DATASETS = {"idrid"}
EXTERNAL_SMDG_PREFIXES = ("PAPILA", "RIM-ONE", "RIMONE")

# при слиянии дубликатов: чьи метки главнее
DATASET_PRIORITY = ["idrid", "odir", "ddr", "aptos", "eyepacs", "smdg", "airogs"]


# ----------------------------------------------------------------------------- utils
def _rec(path, dataset, image_id, patient, eye="", dr_grade=-1, dr=-1, glaucoma=-1, cataract=-1, subset=""):
    dr_grade = int(dr_grade)
    dr = int(dr)
    if dr_grade >= 0 and dr < 0:
        dr = int(dr_grade >= 1)
    return dict(
        src=str(path), dataset=dataset, subset=subset or dataset,
        image_id=f"{dataset}/{image_id}", patient=f"{dataset}/{patient}", eye=str(eye),
        dr_grade=dr_grade, dr=dr, glaucoma=int(glaucoma), cataract=int(cataract),
    )


def list_images(root) -> list[Path]:
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if Path(fn).suffix.lower() in IMG_EXT:
                out.append(Path(dp) / fn)
    return out


def find_files(root, name_regex) -> list[Path]:
    rx = re.compile(name_regex, re.I)
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if rx.search(fn):
                out.append(Path(dp) / fn)
    return sorted(out)


def show_tree(root, max_lines=40):
    """Печатает структуру папок (для отладки, если датасет поменял формат)."""
    root = Path(root)
    lines = 0
    for dp, dns, fns in os.walk(root):
        depth = len(Path(dp).relative_to(root).parts)
        if depth > 3:
            continue
        print("  " * depth + Path(dp).name + f"/  ({len(fns)} files)")
        lines += 1
        if lines >= max_lines:
            print("  ...")
            return


# ----------------------------------------------------------------------------- loaders
def load_eyepacs_aptos(root) -> list[dict]:
    imgs = list_images(root)
    by_stem = {}
    for p in imgs:
        by_stem.setdefault(p.stem.lower(), p)
    recs, miss = [], 0
    for csv in find_files(root, r"^(train|test)Labels15\.csv$"):
        df = pd.read_csv(csv)
        for im, lv in zip(df["image"].astype(str), df["level"]):
            p = by_stem.get(im.lower())
            if p is None:
                miss += 1
                continue
            pid, eye = (im.split("_") + [""])[:2]
            recs.append(_rec(p, "eyepacs", im, pid, eye, dr_grade=lv))
    for csv in find_files(root, r"^trainLabels19\.csv$"):
        df = pd.read_csv(csv)
        for im, lv in zip(df["id_code"].astype(str), df["diagnosis"]):
            p = by_stem.get(im.lower())
            if p is None:
                miss += 1
                continue
            recs.append(_rec(p, "aptos", im, im, "", dr_grade=lv))
    print(f"  eyepacs+aptos: {len(recs)} records, {miss} missing images")
    return recs


def load_ddr(root) -> list[dict]:
    recs, miss, ungradable = [], 0, 0
    txts = [t for t in find_files(root, r"^(train|valid|test)\.txt$") if "grading" in str(t).lower()]
    for txt in txts:
        img_dir = txt.parent / txt.stem
        by_name = {p.name.lower(): p for p in list_images(img_dir)} if img_dir.exists() else {}
        for line in Path(txt).read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            name, grade = parts[0], int(parts[1])
            if grade > 4:  # 5 = ungradable
                ungradable += 1
                continue
            p = by_name.get(name.lower())
            if p is None:
                miss += 1
                continue
            stem = Path(name).stem
            recs.append(_rec(p, "ddr", stem, stem, "", dr_grade=grade))
    print(f"  ddr: {len(recs)} records, {miss} missing, {ungradable} ungradable skipped")
    return recs


def load_idrid(root) -> list[dict]:
    recs, miss = [], 0
    imgs = [p for p in list_images(root) if "grading" in str(p).lower()]
    for csv in find_files(root, r"\.csv$"):
        if "grading" not in str(csv).lower():
            continue
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        if "Image name" not in df.columns or "Retinopathy grade" not in df.columns:
            continue
        df = df.dropna(subset=["Image name"])
        is_train = "train" in csv.name.lower()
        pool = [p for p in imgs if ("train" in str(p.parent).lower()) == is_train]
        by_stem = {p.stem.lower(): p for p in pool}
        tag = "tr" if is_train else "te"
        for im, g in zip(df["Image name"].astype(str), df["Retinopathy grade"]):
            p = by_stem.get(im.strip().lower())
            if p is None:
                miss += 1
                continue
            recs.append(_rec(p, "idrid", f"{tag}_{im}", f"{tag}_{im}", "", dr_grade=int(g)))
    print(f"  idrid: {len(recs)} records, {miss} missing")
    return recs


def _odir_eye_labels(kw: str):
    """Метки одного глаза из ODIR 'Diagnostic Keywords' (а не общие метки пациента)."""
    k = str(kw).lower()
    parts = [x.strip() for x in re.split(r"[，,;]", k) if x.strip()]
    if any("no fundus image" in x for x in parts):
        return None
    grade, dr = 0, 0
    for x in parts:
        if "proliferative" in x or "diabetic retinopathy" in x:
            dr = 1
            if "mild" in x:
                g = 1
            elif "moderate" in x:
                g = 2
            elif "severe" in x and "non" in x:
                g = 3
            elif "proliferative" in x and "non" not in x:
                g = 4
            else:
                g = -1
            grade = g if (grade == 0 or g > grade) else grade
            if g == -1 and grade == 0:
                grade = -1
    glau = 0
    if any("glaucoma" in x and "suspect" not in x for x in parts):
        glau = 1
    elif any("glaucoma" in x for x in parts):
        glau = -1
    cat = int(any("cataract" in x for x in parts))
    return grade, dr, glau, cat


def load_odir(root) -> list[dict]:
    imgs = list_images(root)
    train_imgs = [p for p in imgs if "training" in str(p).lower()] or imgs
    by_name = {}
    for p in train_imgs:
        by_name.setdefault(p.name.lower(), p)
    # data.xlsx (оригинал) или full_df.csv (копия Kaggle)
    df = None
    for f in find_files(root, r"^data\.xlsx$"):
        df = pd.read_excel(f)
        break
    if df is None:
        f = find_files(root, r"^full_df\.csv$")[0]
        df = pd.read_csv(f)
    df = df.drop_duplicates(subset=["ID"])
    recs, miss, skipped = [], 0, 0
    for _, r in df.iterrows():
        for side, eye in (("Left", "left"), ("Right", "right")):
            fn = str(r[f"{side}-Fundus"])
            lab = _odir_eye_labels(r[f"{side}-Diagnostic Keywords"])
            if lab is None:
                skipped += 1
                continue
            p = by_name.get(fn.lower())
            if p is None:
                miss += 1
                continue
            g, d, gl, c = lab
            recs.append(_rec(p, "odir", Path(fn).stem, r["ID"], eye, dr_grade=g, dr=d, glaucoma=gl, cataract=c))
    print(f"  odir: {len(recs)} records (per-eye labels), {miss} missing, {skipped} no-fundus skipped")
    return recs


def load_smdg(root) -> list[dict]:
    meta = None
    for f in find_files(root, r"\.csv$"):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if {"names", "types"} <= set(d.columns):
            meta = d
            break
    assert meta is not None, "SMDG: не найден metadata csv с колонками names/types"
    imgs = [p for p in list_images(root) if "full-fundus" in str(p).lower()]
    by_stem = {p.stem.lower(): p for p in imgs}
    recs, miss = [], 0
    for name, t in zip(meta["names"].astype(str), meta["types"]):
        if Path(name).suffix.lower() in IMG_EXT:
            name = Path(name).stem
        p = by_stem.get(name.lower())
        if p is None:
            miss += 1
            continue
        prefix = re.split(r"-\d|_\d", name)[0]
        g = int(t) if int(t) in (0, 1) else -1  # -1 = подозрение
        recs.append(_rec(p, "smdg", name, name, "", glaucoma=g, subset=f"smdg:{prefix}"))
    c = Counter(r["subset"] for r in recs)
    print(f"  smdg: {len(recs)} records, {miss} missing; sources: {dict(c.most_common())}")
    return recs


def load_airogs(root) -> list[dict]:
    recs = []
    for p in list_images(root):
        lab = p.parent.name.upper()
        if lab not in ("RG", "NRG"):
            continue
        recs.append(_rec(p, "airogs", p.stem, p.stem, "", glaucoma=int(lab == "RG")))
    print(f"  airogs: {len(recs)} records")
    return recs


LOADERS = {
    "eyepacs_aptos": load_eyepacs_aptos,
    "ddr": load_ddr,
    "idrid": load_idrid,
    "odir": load_odir,
    "smdg": load_smdg,
    "airogs": load_airogs,
}


# ----------------------------------------------------------------------------- image processing
def crop_fundus(img: np.ndarray, size: int = IMG_SIZE):
    """Обрезка чёрных полей + центрирование круга сетчатки в квадрат size x size."""
    if img is None or img.size == 0:
        return None
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    thr = max(10, float(np.percentile(gray, 5)) + 8)
    mask = gray > thr
    rows = np.where(mask.mean(1) > 0.02)[0]
    cols = np.where(mask.mean(0) > 0.02)[0]
    if len(rows) < h * 0.2 or len(cols) < w * 0.2:
        y0, y1, x0, x1 = 0, h, 0, w  # маска не нашлась — берём целиком
    else:
        y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    img = img[y0:y1, x0:x1]
    h, w = img.shape[:2]
    s = max(h, w)
    canvas = np.zeros((s, s, 3), dtype=img.dtype)
    canvas[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def fingerprint(img512: np.ndarray):
    """Перцептивный хэш центральной части (без чёрного круга) + маленькая high-pass миниатюра."""
    g = cv2.cvtColor(img512, cv2.COLOR_BGR2GRAY).astype(np.float32)
    s = g.shape[0]
    c = g[int(s * 0.2):int(s * 0.8), int(s * 0.2):int(s * 0.8)]
    small = cv2.resize(c, (64, 64), interpolation=cv2.INTER_AREA)
    d = cv2.dct(small)[:8, :8].flatten()
    med = np.median(d[1:])
    d[0] = med  # DC-компоненту (общая яркость) не учитываем
    h = 0
    for b in (d > med):
        h = (h << 1) | int(b)
    # миниатюра сосудистого рисунка: high-pass + обрезка выбросов (чтобы яркий диск не доминировал)
    c2 = g[int(s * 0.15):int(s * 0.85), int(s * 0.15):int(s * 0.85)]
    t = cv2.resize(c2, (48, 48), interpolation=cv2.INTER_AREA)
    t = t - cv2.GaussianBlur(t, (0, 0), 1.5)
    t = np.clip((t - t.mean()) / (t.std() + 1e-6), -1.5, 1.5)
    return np.uint64(h), t.astype(np.float16)


def _process_one(args):
    src, dst = args
    try:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if max(h, w) > 2 * IMG_SIZE:  # ускоряем: сначала грубо уменьшаем
            f = 2 * IMG_SIZE / max(h, w)
            img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
        out = crop_fundus(img)
        if out is None:
            return None
        if out.mean() < 8:  # пустой/чёрный кадр
            return None
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        ph, th = fingerprint(out)
        return int(ph), th
    except Exception:
        return None


def process_images(recs: list[dict], out_dir: str, workers: int | None = None) -> pd.DataFrame:
    workers = workers or os.cpu_count() or 2
    jobs = []
    for r in recs:
        rel = f"img/{r['image_id']}.jpg".replace(":", "_")
        r["path"] = rel
        jobs.append((r["src"], os.path.join(out_dir, rel)))
    keep, hashes, thumbs = [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for r, res in tqdm(zip(recs, ex.map(_process_one, jobs, chunksize=32)), total=len(jobs), desc="images"):
            if res is None:
                continue
            keep.append(r)
            hashes.append(res[0])
            thumbs.append(res[1])
    df = pd.DataFrame(keep).drop(columns=["src"])
    df["phash"] = np.array(hashes, dtype=np.uint64).astype(np.int64)  # int64 для csv
    print(f"  processed {len(df)}/{len(recs)} images")
    return df, (np.stack(thumbs) if thumbs else np.zeros((0, 48, 48), np.float16))


# ----------------------------------------------------------------------------- dedup + split
class DSU:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def popcount64(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64)
    if hasattr(np, "bitwise_count"):  # numpy >= 2.0
        return np.bitwise_count(x).astype(np.int64)
    c = np.zeros(x.shape, dtype=np.int64)
    for i in range(8):
        c += np.unpackbits(((x >> np.uint64(8 * i)) & np.uint64(255)).astype(np.uint8)[..., None], axis=-1).sum(-1)
    return c


def find_duplicates(phash: np.ndarray, thumbs: np.ndarray, max_ham=10, min_corr=0.85, max_bucket=800,
                    chunk=20000):
    """Кандидаты — совпадение одного из 4 кусков по 16 бит (гарантирует нахождение пар с hamming<=3,
    большинство пар до ~8), подтверждение — корреляция high-pass миниатюр.
    Пары проверяются кусками, чтобы не съесть всю RAM Colab."""
    ph = phash.astype(np.uint64)
    n = len(ph)
    flat = thumbs.reshape(n, -1)  # float16, ~0.6 GB на 130k снимков
    norm = np.sqrt(np.einsum("ij,ij->i", flat, flat, dtype=np.float32)) + 1e-6
    pairs = set()
    big, cand = 0, 0
    for k in range(4):
        key = ((ph >> np.uint64(16 * k)) & np.uint64(0xFFFF)).astype(np.int64)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], len(ks)]
        for s, e in zip(starts, ends):
            m = e - s
            if m < 2:
                continue
            if m > max_bucket:
                big += 1
                continue
            idx = order[s:e]
            x = ph[idx]
            ham = popcount64(x[:, None] ^ x[None, :])
            ii, jj = np.where(np.triu(ham <= max_ham, 1))
            cand += len(ii)
            for c in range(0, len(ii), chunk):
                a, b = idx[ii[c:c + chunk]], idx[jj[c:c + chunk]]
                corr = np.einsum("ij,ij->i", flat[a], flat[b], dtype=np.float32) / (norm[a] * norm[b])
                ok = corr >= min_corr
                for u, v in zip(a[ok], b[ok]):
                    pairs.add((int(min(u, v)), int(max(u, v))))
    print(f"  кандидатов проверено: {cand}, пропущено больших корзин: {big}")
    return sorted(pairs)


def merge_and_split(df: pd.DataFrame, thumbs: np.ndarray, seed=2026, val_frac=0.12, test_frac=0.10) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    n = len(df)
    pairs = find_duplicates(df["phash"].values.astype(np.int64).view(np.uint64), thumbs)
    print(f"  near-duplicate pairs: {len(pairs)}")
    dsn = df["dataset"].values
    print("  пары дубликатов по датасетам:", dict(Counter(tuple(sorted((dsn[a], dsn[b]))) for a, b in pairs).most_common(12)))

    # 1) кластеры дубликатов → оставляем один снимок, сливаем метки
    dup = DSU(n)
    for a, b in pairs:
        dup.union(a, b)
    droot = np.array([dup.find(i) for i in range(n)])
    prio = {d: i for i, d in enumerate(DATASET_PRIORITY)}
    df["_prio"] = df["dataset"].map(lambda d: prio.get(d, 99))
    df["_ext"] = is_external(df)
    sizes = np.bincount(droot, minlength=n)
    keep_mask = sizes[droot] == 1
    removed = 0
    multi = defaultdict(list)
    for i in np.flatnonzero(sizes[droot] > 1):
        multi[droot[i]].append(i)
    for root, idxs in multi.items():
        sub = df.loc[idxs].sort_values(["_ext", "_prio"], ascending=[False, True])
        k = sub.index[0]
        for col in LABEL_COLS:
            if df.at[k, col] >= 0:
                continue
            vals = [v for v in sub[col].tolist() if v >= 0]
            if vals:
                df.at[k, col] = vals[0] if (len(set(vals)) == 1 or col == "dr_grade") else -1
        keep_mask[k] = True
        removed += len(idxs) - 1
    keep_idx = np.flatnonzero(keep_mask)
    fix = (df["dr"] < 0) & (df["dr_grade"] >= 0)
    df.loc[fix, "dr"] = (df.loc[fix, "dr_grade"] >= 1).astype(int)
    print(f"  removed {removed} duplicate images across/within datasets")

    # 2) группы для сплита = пациенты ∪ кластеры дубликатов
    g = DSU(n)
    for a, b in pairs:
        g.union(a, b)
    first_of_patient = {}
    for i, p in enumerate(df["patient"].values):
        if p in first_of_patient:
            g.union(i, first_of_patient[p])
        else:
            first_of_patient[p] = i
    groot = np.array([g.find(i) for i in range(n)])
    ext_groups = set(groot[df["_ext"].values])

    def bucket(root):
        h = int(hashlib.md5(f"{seed}-{root}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return "test" if h < test_frac else ("val" if h < test_frac + val_frac else "train")

    df["split"] = [("test_ext" if r in ext_groups else bucket(r)) for r in groot]
    df["group"] = groot
    out = df.loc[sorted(keep_idx)].drop(columns=["_prio", "_ext"]).reset_index(drop=True)
    return out


def is_external(df: pd.DataFrame) -> np.ndarray:
    ext = df["dataset"].isin(EXTERNAL_DATASETS).values.copy()
    sub = df["subset"].astype(str).str.upper()
    for pfx in EXTERNAL_SMDG_PREFIXES:
        ext |= sub.str.startswith(f"SMDG:{pfx}").values
    return ext


def summarize(df: pd.DataFrame):
    print("\n=== Итоговый датасет ===")
    print(df.groupby(["split", "dataset"]).size().unstack(0).fillna(0).astype(int))
    for col in ["dr", "glaucoma", "cataract"]:
        t = df[df[col] >= 0].groupby("split")[col].agg(["count", "sum"])
        print(f"\n{col}: (count = размечено, sum = больных)\n{t}")
    t = df[df["dr_grade"] >= 0].groupby(["split", "dr_grade"]).size().unstack(1).fillna(0).astype(int)
    print(f"\ndr_grade:\n{t}")


# ----------------------------------------------------------------------------- pipeline
def run_all(work_dir="/content/veya_data", drive_dir="/content/drive/MyDrive/veya_v2/data",
            only: list[str] | None = None, delete_raw=True):
    """Каждый датасет после обработки сразу сохраняется на Drive (parts/<name>.tar + .csv + thumbs),
    поэтому если Colab оборвётся — при перезапуске готовые датасеты пропускаются."""
    import kagglehub

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    parts = Path(drive_dir) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    failed = []

    for key, slug in SOURCES.items():
        if only and key not in only:
            continue
        if (parts / f"{key}.csv").exists():
            print(f"[{key}] уже на Drive, пропускаю")
            continue
        t0 = time.time()
        print(f"\n[{key}] скачиваю {slug} ...")
        try:
            root = kagglehub.dataset_download(slug)
            print(f"[{key}] скачано за {time.time()-t0:.0f}s")
            recs = LOADERS[key](root)
            if not recs:
                raise RuntimeError("0 записей")
        except Exception as e:
            print(f"[{key}] ОШИБКА: {e!r}\nСтруктура папки:")
            try:
                show_tree(root)
            except Exception:
                pass
            failed.append(key)
            continue
        df, thumbs = process_images(recs, str(work))
        # на Drive: tar картинок этого датасета + метки + отпечатки
        tmp = work / f"{key}.tar"
        with tarfile.open(tmp, "w") as tar:
            for p in df["path"]:
                tar.add(work / p, arcname=p)
        shutil.copy(tmp, parts / f"{key}.tar")
        os.remove(tmp)
        np.save(parts / f"{key}_thumbs.npy", thumbs)
        df.to_csv(parts / f"{key}.csv", index=False)  # csv последним = маркер «готово»
        if delete_raw:
            shutil.rmtree(root, ignore_errors=True)
            shutil.rmtree(Path.home() / ".cache" / "kagglehub", ignore_errors=True)
        print(f"[{key}] готово за {(time.time()-t0)/60:.1f} мин, {len(df)} снимков")

    if failed:
        print(f"\n⚠ Не получилось: {failed}. Остальное сохранено; можно повторить run_all(only={failed}).")
    return build_labels(drive_dir)


def build_labels(drive_dir="/content/drive/MyDrive/veya_v2/data"):
    """Склейка всех датасетов → дедупликация → сплит по пациентам → labels.csv на Drive."""
    parts = Path(drive_dir) / "parts"
    keys = [k for k in SOURCES if (parts / f"{k}.csv").exists()]
    df = pd.concat([pd.read_csv(parts / f"{k}.csv") for k in keys], ignore_index=True)
    thumbs = np.concatenate([np.load(parts / f"{k}_thumbs.npy") for k in keys])
    final = merge_and_split(df, thumbs)
    final.to_csv(Path(drive_dir) / "labels.csv", index=False)
    summarize(final)
    return final


def restore_from_drive(local_dir="/content/veya_data", drive_dir="/content/drive/MyDrive/veya_v2/data"):
    """Новая сессия Colab: распаковываем tar-ы с Drive на быстрый локальный диск."""
    local = Path(local_dir)
    local.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for t in sorted((Path(drive_dir) / "parts").glob("*.tar")):
        flag = local / f".{t.stem}.done"
        if flag.exists():
            continue
        with tarfile.open(t) as tar:
            tar.extractall(local)
        flag.touch()
        print(f"  распакован {t.name}")
    shutil.copy(Path(drive_dir) / "labels.csv", local / "labels.csv")
    print(f"Данные готовы за {time.time()-t0:.0f}s")
    return local
