"""Explanations and recommendations for VEYA v2 answers.

* Grad-CAM in numpy: the ONNX graph returns the last feature map ("feat", un-flipped TTA branch);
  cam_head.npz holds the neck/head weights, so the gradient of each disease logit with respect to
  the pooled features is computed analytically (Linear -> GELU -> Linear -> GELU -> head).
* Anatomical landmarks (optic disc, macula) are located with simple image heuristics, and the share
  of the heat map that falls on each zone is turned into a plain-language justification.
* Image quality (brightness, contrast, sharpness) is checked, because a blurred or dark photo makes
  any AI answer less reliable.
* Recommendations are tiered by urgency. They are screening guidance, not a diagnosis.
"""
from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_HEAD = None
BINARY = ("dr", "glaucoma", "cataract")


# ----------------------------------------------------------------------------- Grad-CAM
def _erf(x):
    # Abramowitz–Stegun 7.1.26, |error| < 1.5e-7 — avoids a scipy dependency
    s = np.sign(x)
    x = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def _gelu_and_grad(x):
    cdf = 0.5 * (1.0 + _erf(x / np.sqrt(2.0)))
    pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    return x * cdf, cdf + x * pdf


def load_head(bundle_dir: Path):
    global _HEAD
    if _HEAD is None:
        f = Path(bundle_dir) / "cam_head.npz"
        _HEAD = {k: v.astype(np.float64) for k, v in np.load(f).items()} if f.exists() else {}
    return _HEAD


def gradcam(feat: np.ndarray, head: dict, key: str, cls: int | None = None) -> np.ndarray:
    """feat: (1, C, h, w). Returns a (h, w) map in [0, 1]."""
    A = feat[0].astype(np.float64)
    f = A.mean(axis=(1, 2))
    h1 = head["neck_1_weight"] @ f + head["neck_1_bias"]
    g1, d1 = _gelu_and_grad(h1)
    h2 = head["neck_4_weight"] @ g1 + head["neck_4_bias"]
    _, d2 = _gelu_and_grad(h2)
    w = head[f"heads_{key}_weight"][0 if cls is None else cls]
    grad_f = head["neck_1_weight"].T @ (d1 * (head["neck_4_weight"].T @ (d2 * w)))
    cam = np.maximum(np.tensordot(grad_f, A, axes=(0, 0)), 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


# ----------------------------------------------------------------------------- image analysis
def retina_mask(img: np.ndarray) -> np.ndarray:
    gray = img @ np.array([0.299, 0.587, 0.114])
    return gray > max(12.0, float(np.percentile(gray, 3)) + 8)


def landmarks(img: np.ndarray, mask: np.ndarray) -> dict:
    """Optic disc = brightest large blob inside the retina; macula ≈ 2.5 disc diameters towards the centre."""
    s = img.shape[0]
    ys, xs = np.where(mask)
    if len(ys) < 100:
        return {}
    cy, cx = ys.mean(), xs.mean()
    radius = np.sqrt(mask.sum() / np.pi)
    lum = Image.fromarray((img[..., 0] * 0.5 + img[..., 1] * 0.5).clip(0, 255).astype(np.uint8))
    lum = np.asarray(lum.filter(ImageFilter.GaussianBlur(s / 40)), dtype=np.float64)
    lum[~mask] = 0
    # keep away from the very rim, where flash reflections live
    yy, xx = np.mgrid[0:s, 0:s]
    lum[np.hypot(yy - cy, xx - cx) > radius * 0.85] = 0
    oy, ox = np.unravel_index(int(np.argmax(lum)), lum.shape)
    od_r = radius * 0.16  # optic disc ≈ 1/6 of the 45° field radius (generous to cover the rim)
    vy, vx = cy - oy, cx - ox
    n = np.hypot(vy, vx) or 1.0
    dist = min(2.5 * 2 * od_r * 0.8, radius * 0.75)
    my, mx = oy + vy / n * dist, ox + vx / n * dist
    return dict(center=(cy, cx), radius=radius, od=(oy, ox), od_r=od_r, macula=(my, mx), mac_r=od_r * 1.3)


def zone_shares(cam_full: np.ndarray, mask: np.ndarray, lm: dict) -> dict:
    s = cam_full.shape[0]
    yy, xx = np.mgrid[0:s, 0:s]
    w = cam_full * mask
    tot = w.sum()
    if tot <= 0 or not lm:
        return {}
    od = np.hypot(yy - lm["od"][0], xx - lm["od"][1]) <= lm["od_r"]
    mac = np.hypot(yy - lm["macula"][0], xx - lm["macula"][1]) <= lm["mac_r"]
    central = (np.hypot(yy - lm["center"][0], xx - lm["center"][1]) <= lm["radius"] * 0.55) & ~od & ~mac
    periph = mask & ~od & ~mac & ~central
    area = mask.sum()
    out = {}
    for name, z in (("optic_disc", od), ("macula", mac), ("central", central), ("periphery", periph)):
        share = float(w[z].sum() / tot)
        out[name] = {"share": round(share, 3), "area_share": round(float((z & mask).sum() / area), 3)}
    # how concentrated the attention is: share of heat in the hottest 10% of the retina
    v = np.sort(w[mask])[::-1]
    out["focus"] = round(float(v[: max(1, len(v) // 10)].sum() / tot), 3)
    return out


def image_quality(img: np.ndarray, mask: np.ndarray) -> dict:
    g = img[..., 1]
    inside = g[mask]
    brightness = float(inside.mean()) if inside.size else 0.0
    contrast = float(inside.std()) if inside.size else 0.0
    lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    sharp = float(lap[mask[1:-1, 1:-1]].mean()) if mask.any() else 0.0
    coverage = float(mask.mean())
    issues = []
    if brightness < 45:
        issues.append("снимок тёмный")
    if brightness > 200:
        issues.append("снимок пересвечен")
    if contrast < 14:
        issues.append("низкий контраст")
    if sharp < 1.6:
        issues.append("снимок размыт")
    if coverage < 0.35:
        issues.append("сетчатка занимает малую часть кадра")
    return {"brightness": round(brightness, 1), "contrast": round(contrast, 1), "sharpness": round(sharp, 2),
            "retina_coverage": round(coverage, 2), "issues": issues,
            "ok": len(issues) == 0}


# ----------------------------------------------------------------------------- overlay
_ANCH = np.array([[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66],
                  [245, 125, 21], [250, 193, 39], [252, 255, 164]], dtype=np.float64)  # inferno


def _inferno(v):
    v = np.clip(v, 0, 1) * (len(_ANCH) - 1)
    i = np.floor(v).astype(int).clip(0, len(_ANCH) - 2)
    t = (v - i)[..., None]
    return _ANCH[i] * (1 - t) + _ANCH[i + 1] * t


def overlay_png(img: np.ndarray, cam_full: np.ndarray, mask: np.ndarray, lm: dict, size=384) -> str:
    heat = _inferno(cam_full)
    a = (cam_full ** 0.8 * 0.75)[..., None] * mask[..., None]
    out = img * 0.6 * (1 - a) + heat * a + img * 0.4 * (1 - a)
    out = Image.fromarray(out.clip(0, 255).astype(np.uint8)).resize((size, size), Image.BILINEAR)
    if lm:
        k = size / img.shape[0]
        d = ImageDraw.Draw(out)
        for (y, x), r, col in ((lm["od"], lm["od_r"], (120, 220, 255)), (lm["macula"], lm["mac_r"], (160, 255, 190))):
            d.ellipse([(x - r) * k, (y - r) * k, (x + r) * k, (y + r) * k], outline=col, width=2)
    buf = BytesIO()
    out.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ----------------------------------------------------------------------------- texts
DR_STAGE_TEXT = {
    0: "признаков ретинопатии нет",
    1: "начальная непролиферативная стадия: по шкале ICDR ей соответствуют только микроаневризмы",
    2: "умеренная непролиферативная стадия: микроаневризмы, точечные кровоизлияния, твёрдые экссудаты",
    3: "тяжёлая непролиферативная стадия: множественные кровоизлияния во всех квадрантах, изменения вен",
    4: "пролиферативная стадия: образуются новые хрупкие сосуды, есть риск кровоизлияния в стекловидное тело",
}
ZONE_RU = {"optic_disc": "диск зрительного нерва", "macula": "макула (зона центрального зрения)",
           "central": "центральная часть сетчатки и сосудистые аркады", "periphery": "периферия сетчатки"}


def _top_zones(z: dict, n=2):
    items = [(k, v["share"], v["area_share"]) for k, v in z.items() if k in ZONE_RU]
    items.sort(key=lambda t: -t[1])
    return items[:n]


def disease_findings(key: str, scr: dict, zones: dict, quality: dict) -> list[str]:
    st = scr[key]["status"]
    p = scr[key]["probability"]
    out = []
    if key == "dr":
        stage = scr["dr_stage"]["class"]
        sp = scr["dr_stage"]["probabilities"][stage]
        if st != "negative":
            out.append(f"Оценка стадии: {stage} из 4 ({DR_STAGE_TEXT[stage]}), уверенность {sp:.0%}.")
        if zones and st != "negative":
            top = _top_zones(zones)
            zs = ", ".join(f"{ZONE_RU[k]} — {s:.0%} внимания" for k, s, _ in top)
            out.append(f"Модель опиралась в основном на: {zs}.")
            if zones["macula"]["share"] > 2 * max(zones["macula"]["area_share"], 0.02) and st != "negative":
                out.append("Подозрительные участки затрагивают макулу. Изменения в этой зоне могут "
                           "снижать центральное зрение, поэтому важна оценка на отёк (ОКТ).")
            if zones["focus"] > 0.45 and st != "negative":
                out.append("Внимание сосредоточено на отдельных очагах: так обычно выглядят локальные "
                           "изменения (кровоизлияния, экссудаты), а не общий фон снимка.")
    elif key == "glaucoma":
        if zones:
            od = zones["optic_disc"]["share"]
            if st != "negative":
                if od >= 0.25:
                    out.append(f"{od:.0%} внимания модели приходится на диск зрительного нерва. Там врач оценивает "
                               "экскавацию (соотношение чаши и диска) и истончение нейроретинального пояска.")
                else:
                    out.append(f"На диск зрительного нерва приходится лишь {od:.0%} внимания, модель реагировала "
                               "и на другие зоны. Такой результат обязательно проверяет врач.")
            else:
                out.append(f"Диск зрительного нерва просмотрен ({od:.0%} внимания), типичных для глаукомы "
                           "изменений модель не выделила.")
    elif key == "cataract":
        if st != "negative":
            out.append("Снимок выглядит мутным, с размытыми деталями сосудов. На фото глазного дна так "
                       "проявляется помутнение хрусталика, но похожий эффект даёт и плохая фокусировка.")
        if not quality["ok"] and st != "negative":
            out.append("Качество снимка снижено (" + ", ".join(quality["issues"]) + "), это может влиять на оценку.")
    if st == "uncertain":
        out.append(f"Вероятность {p:.0%} попала в «зону сомнения»: ИИ не даёт уверенного ответа и передаёт решение врачу.")
    return out


def recommendations(scr: dict, quality: dict) -> dict:
    """Tiered, screening-level guidance. Timeframes follow common referral practice for screening programmes."""
    items = []  # (level, title, timeframe, actions)
    dr, gl, ca = scr["dr"]["status"], scr["glaucoma"]["status"], scr["cataract"]["status"]
    stage = scr["dr_stage"]["class"]

    if dr == "positive" and stage == 4:
        items.append(("urgent", "Признаки пролиферативной ретинопатии", "в течение 1–3 дней",
                      ["Срочно запишитесь к офтальмологу (ретинологу): обычно нужна лазерная коагуляция или инъекции в глаз.",
                       "Не откладывайте: новые сосуды могут кровоточить и вызвать резкую потерю зрения.",
                       "Сообщите лечащему врачу (терапевту или эндокринологу) о результате."]))
    elif dr == "positive" and stage == 3:
        items.append(("high", "Признаки тяжёлой непролиферативной ретинопатии", "в течение 1–2 недель",
                      ["Запишитесь к офтальмологу: осмотр глазного дна с расширением зрачка и ОКТ макулы.",
                       "Высокий риск перехода в пролиферативную стадию, нужен план лечения.",
                       "Обсудите с лечащим врачом контроль уровня глюкозы и давления."]))
    elif dr == "positive":
        items.append(("medium", "Признаки ретинопатии", "в течение 1 месяца",
                      ["Плановый осмотр офтальмолога с расширением зрачка, при необходимости ОКТ.",
                       "Обсудите с лечащим врачом контроль глюкозы, давления и холестерина.",
                       "Повторный скрининг — через 6 месяцев или по назначению врача."]))
    if gl == "positive":
        items.append(("high", "Подозрение на глаукому", "в течение 2 недель",
                      ["Офтальмолог: измерение внутриглазного давления, осмотр диска зрительного нерва, периметрия (поля зрения), ОКТ.",
                       "Глаукома долго протекает без симптомов, а потерянное зрение не восстанавливается. Не откладывайте визит.",
                       "Сообщите врачу, если глаукома была у родственников."]))
    if ca == "positive":
        items.append(("low", "Признаки катаракты", "в течение 1–3 месяцев",
                      ["Плановый осмотр офтальмолога (биомикроскопия хрусталика).",
                       "Если ухудшение мешает чтению или вождению, запишитесь раньше: катаракта хорошо лечится операцией."]))
    unc = [scr[k]["name_ru"] for k in BINARY if scr[k]["status"] == "uncertain"]
    if unc:
        items.append(("medium", "ИИ не уверен: " + ", ".join(unc), "в течение 1 месяца",
                      ["Покажите снимок офтальмологу или пройдите очный осмотр.",
                       "Если есть возможность, переснимите глаз при хорошей фокусировке и широком зрачке."]))
    if not quality["ok"]:
        items.append(("info", "Качество снимка", "сейчас",
                      ["Переснимите: " + ", ".join(quality["issues"]) + ". Хороший снимок повышает точность ИИ.",
                       "Затемните помещение, держите насадку неподвижно, сетчатка должна заполнять кадр."]))
    if not items or all(i[0] == "info" for i in items):
        items.append(("routine", "Признаков заболеваний не найдено", "через 12 месяцев",
                      ["Плановый повторный скрининг через год.",
                       "Если есть диабет или повышенное давление, осмотр раз в год обязателен."]))

    red_flags = ["Резкое снижение зрения, «шторка» перед глазом, вспышки или много новых «мушек».",
                 "Сильная боль в глазу, покраснение, радужные круги вокруг источников света, тошнота."]
    order = {"urgent": 0, "high": 1, "medium": 2, "low": 3, "routine": 4, "info": 5}
    items.sort(key=lambda t: order[t[0]])
    level_ru = {"urgent": "срочно", "high": "высокий приоритет", "medium": "в плановом порядке, но не откладывать",
                "low": "планово", "routine": "профилактика", "info": "к сведению"}
    top = items[0]
    return {
        "urgency": top[0],
        "urgency_ru": level_ru[top[0]],
        "summary_ru": f"{top[1]}: обратиться к офтальмологу {top[2]}." if top[0] not in ("routine", "info")
                      else f"{top[1]}. Повторная проверка {top[2]}.",
        "items": [{"level": l, "level_ru": level_ru[l], "title": t, "timeframe": tf, "actions": a} for l, t, tf, a in items],
        "red_flags": {"title": "Немедленно к офтальмологу или в неотложную помощь (в течение 24 часов), если появилось:",
                      "items": red_flags},
    }


# ----------------------------------------------------------------------------- main entry
def explain(img_chw: np.ndarray, feat: np.ndarray | None, screening: dict, bundle_dir: Path) -> dict:
    img = img_chw[0].transpose(1, 2, 0).astype(np.float64)  # (s, s, 3) RGB 0..255
    s = img.shape[0]
    mask = retina_mask(img)
    lm = landmarks(img, mask)
    quality = image_quality(img, mask)
    head = load_head(bundle_dir)

    order = sorted(BINARY, key=lambda k: ({"positive": 0, "uncertain": 1, "negative": 2}[screening[k]["status"]],
                                          -screening[k]["probability"]))
    per = {}
    heatmap = None
    for k in BINARY:
        zones = {}
        if feat is not None and head:
            cam = gradcam(feat, head, k)
            cam_full = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((s, s), Image.BICUBIC),
                                  dtype=np.float64) / 255.0
            zones = zone_shares(cam_full, mask, lm)
            if k == order[0]:
                heatmap = {"disease": k, "name_ru": screening[k]["name_ru"],
                           "image": overlay_png(img, cam_full, mask, lm),
                           "legend_ru": "Яркие участки — то, на что опиралась нейросеть. Голубой круг — диск "
                                        "зрительного нерва, зелёный — макула (найдены автоматически)."}
        per[k] = {"name_ru": screening[k]["name_ru"], "status": screening[k]["status"],
                  "attention_zones": zones, "findings": disease_findings(k, screening, zones, quality)}

    main = order[0]
    st = screening[main]["status"]
    if st == "negative":
        reasoning = ("Нейросеть просмотрела диск зрительного нерва, макулу и сосуды и не нашла признаков, "
                     "типичных для ретинопатии, глаукомы или катаракты.")
    else:
        f = per[main]["findings"]
        key_line = f[1:2] if main == "dr" else f[:1]  # the stage line is already shown next to the probabilities
        reasoning = f"Главная находка: {screening[main]['name_ru'].lower()} ({screening[main]['status_ru']}). " + " ".join(key_line)
    return {"reasoning_ru": reasoning, "heatmap": heatmap, "diseases": per, "image_quality": quality,
            "landmarks_found": bool(lm),
            "method_ru": "Grad-CAM: по градиентам нейросети видно, какие участки снимка повлияли на ответ."}
