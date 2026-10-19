"""VEYA v2 — explained answers: where the model looked, which changes it relied on, what to do next.

Everything here runs on NumPy + Pillow next to the ONNX model (no PyTorch):

* Grad-CAM. The ONNX graph also returns the last backbone feature maps of both TTA passes
  (`feat`, `feat_flip`, added by tools/add_explain_outputs.py). The head on top of them is a small
  MLP (pool → 1280→768 → 768→384 → disease), so the gradient of each disease logit with respect to
  the pooled features is computed analytically from the weights in model_bundle/explain_head.npz.
* Retina anatomy. Optic disc = brightest smoothed spot inside the retina; macula = darkest smoothed
  spot about 2.5 disc diameters towards the image centre. Attention is then reported per region.
* Hotspots. Separate peaks of the attention map are described by sector, region and distance from the
  macula in disc diameters (≤ 1 DD from the fovea = risk of macular oedema, as in ICDR grading).
  Vessel contrast describes lens opacity (cataract) and image quality.
* Recommendations by urgency (ICDR stages for DR), with timeframes, exams and red-flag symptoms.

The explanation shows what the AI answer is based on; it is not a lesion-by-lesion diagnosis.
"""
from __future__ import annotations

import base64
import math
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

DISEASES = ("dr", "glaucoma", "cataract")
LANGS = ("ru", "en", "kk")


# ============================================================================ Grad-CAM
class CamHead:
    def __init__(self, path: Path):
        w = np.load(path)
        self.w1, self.b1 = w["neck_1_weight"], w["neck_1_bias"]
        self.w2, self.b2 = w["neck_4_weight"], w["neck_4_bias"]
        self.heads = {k: (w[f"heads_{k}_weight"], w[f"heads_{k}_bias"]) for k in ("dr_grade",) + DISEASES}

    @staticmethod
    def _gelu_grad(z):
        erf = np.vectorize(math.erf)(z / math.sqrt(2.0))
        return 0.5 * (1.0 + erf) + z * np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    def grad(self, f: np.ndarray, head: str, cls: int | None = None) -> np.ndarray:
        """d(logit)/d(pooled features) for one image, f: (C,)."""
        z1 = self.w1 @ f + self.b1
        h1 = 0.5 * z1 * (1.0 + np.vectorize(math.erf)(z1 / math.sqrt(2.0)))
        z2 = self.w2 @ h1 + self.b2
        w, _ = self.heads[head]
        out = w[cls] if cls is not None else w[0]
        g2 = out * self._gelu_grad(z2)
        g1 = (self.w2.T @ g2) * self._gelu_grad(z1)
        return self.w1.T @ g1

    def cam(self, feat: np.ndarray, feat_flip: np.ndarray, head: str, cls: int | None = None) -> np.ndarray:
        """Grad-CAM averaged over both TTA passes, (H, W) in 0..1."""
        maps = []
        for a, flipped in ((feat[0], False), (feat_flip[0], True)):
            g = self.grad(a.mean((1, 2)), head, cls)
            m = np.maximum(np.tensordot(g, a, axes=1), 0.0)
            maps.append(m[:, ::-1] if flipped else m)
        m = (maps[0] + maps[1]) / 2
        return m / m.max() if m.max() > 0 else m


def upsample(cam: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray((cam * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


# ============================================================================ anatomy
def _blur(a: np.ndarray, r: float) -> np.ndarray:
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(r)), dtype=np.float32)


def _erode(mask: np.ndarray, px: int) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(2 * px + 1))
    return np.asarray(img) > 127


def anatomy(rgb: np.ndarray) -> dict:
    """Retina mask, optic disc and macula on the model's 512×512 input."""
    s = rgb.shape[0]
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    mask = _blur(gray, 2) > max(12.0, float(np.percentile(gray, 3)) + 8)
    ys, xs = np.nonzero(mask)
    if len(ys) < s * s * 0.2:
        mask = np.ones_like(mask)
        ys, xs = np.nonzero(mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    radius = float(np.sqrt(mask.sum() / np.pi))
    inner = _erode(mask, max(4, s // 40))

    # optic disc: brightest large blob (red+green, blurred at disc scale)
    lum = _blur(0.5 * rgb[..., 0] + 0.5 * rgb[..., 1], s / 40)
    lum_in = np.where(inner, lum, -1)
    dy, dx = np.unravel_index(int(np.argmax(lum_in)), lum_in.shape)
    disc_r = 0.09 * radius  # disc diameter ≈ 1/5.5 of a 45° field

    # macula: darkest blurred spot 2–3 disc diameters from the disc, towards the centre
    vx, vy = cx - dx, cy - dy
    norm = math.hypot(vx, vy) or 1.0
    vx, vy = vx / norm, vy / norm
    dist = 2.5 * 2 * disc_r
    ex, ey = dx + vx * dist, dy + vy * dist
    dark = _blur(gray, s / 50)
    yy, xx = np.mgrid[0:s, 0:s]
    win = inner & ((yy - ey) ** 2 + (xx - ex) ** 2 < (1.4 * disc_r * 2) ** 2)
    if win.any():
        my, mx = np.unravel_index(int(np.argmin(np.where(win, dark, 1e9))), dark.shape)
    else:
        my, mx = int(ey), int(ex)
    return dict(mask=mask, inner=inner, center=(cy, cx), radius=radius,
                disc=(int(dy), int(dx)), disc_r=disc_r, macula=(int(my), int(mx)), gray=gray)


def region_masks(an: dict, s: int) -> dict:
    yy, xx = np.mgrid[0:s, 0:s]
    dy, dx = an["disc"]
    my, mx = an["macula"]
    cy, cx = an["center"]
    d_disc = np.hypot(yy - dy, xx - dx)
    d_mac = np.hypot(yy - my, xx - mx)
    d_c = np.hypot(yy - cy, xx - cx)
    mask = an["mask"]
    disc = mask & (d_disc < 2.2 * an["disc_r"])  # disc + peripapillary zone
    macula = mask & ~disc & (d_mac < 2.0 * an["disc_r"])
    periphery = mask & ~disc & ~macula & (d_c > 0.72 * an["radius"])
    posterior = mask & ~disc & ~macula & ~periphery
    return {"optic_disc": disc, "macula": macula, "vascular_arcades": posterior, "periphery": periphery}


def quadrant(an: dict, y: float, x: float) -> str:
    """superior/inferior + temporal (macula side) / nasal (disc side)."""
    dy, dx = an["disc"]
    my, mx = an["macula"]
    vert = "superior" if y < (dy + my) / 2 else "inferior"
    temporal = (x - dx) * (mx - dx) + (y - dy) * (my - dy) > 0
    return f"{vert}_{'temporal' if temporal else 'nasal'}"


# ============================================================================ attention hotspots
def hotspots(att: np.ndarray, an: dict, k: int = 3, min_level: float = 0.45) -> list[dict]:
    """Separate peaks of the attention map (greedy non-maximum suppression, ≥ 1 disc diameter apart)."""
    s = att.shape[0]
    step = s // 64
    grid = att[step // 2::step, step // 2::step] * an["mask"][step // 2::step, step // 2::step]
    dd = 2 * an["disc_r"]
    regs = region_masks(an, s)
    my, mx = an["macula"]
    picked: list[tuple[int, int]] = []
    out = []
    for flat in np.argsort(-grid, axis=None):
        gy, gx = divmod(int(flat), grid.shape[1])
        if grid[gy, gx] < min_level or len(out) == k:
            break
        y, x = gy * step + step // 2, gx * step + step // 2
        if any(math.hypot(y - py, x - px) < dd for py, px in picked):
            continue
        picked.append((y, x))
        region = next((r for r, m in regs.items() if m[y, x]), "vascular_arcades")
        out.append(dict(y=round(y / s, 3), x=round(x / s, 3), strength=int(round(100 * float(att[y, x]))),
                        region=region, sector=quadrant(an, y, x),
                        macula_dd=round(math.hypot(y - my, x - mx) / dd, 1)))
    return out


def image_quality(rgb: np.ndarray, an: dict) -> dict:
    g = rgb[..., 1]
    inner = an["inner"]
    lap = (np.roll(g, 1, 0) + np.roll(g, -1, 0) + np.roll(g, 1, 1) + np.roll(g, -1, 1) - 4 * g)
    # vessel sharpness: high-frequency energy relative to local brightness
    sharp = float(np.sqrt(np.mean(lap[inner] ** 2)) / (np.mean(g[inner]) + 1.0) * 100)
    vessel = float(np.mean(np.maximum(_blur(g, 6) - _blur(g, 1.2), 0)[inner]) / (np.mean(g[inner]) + 1.0) * 100)
    bright = float(np.mean(rgb[inner]))
    coverage = float(an["mask"].mean())
    issues = []
    if bright < 35:
        issues.append("too_dark")
    if bright > 200:
        issues.append("overexposed")
    if coverage < 0.45:
        issues.append("small_field")
    if vessel < VESSEL_LOW:
        issues.append("blurred")
    return dict(sharpness=round(sharp, 2), vessel_contrast=round(vessel, 2), brightness=round(bright, 1),
                field_coverage=round(coverage, 3), issues=issues,
                good=not issues)


VESSEL_LOW = 0.8  # vessel contrast below this: the fundus looks hazy (clear images ≥ 1.0, cataract 0.45–0.9)


# ============================================================================ texts
T = {
    "regions": {
        "optic_disc": {"ru": "диск зрительного нерва", "en": "optic disc", "kk": "көру нервінің дискі"},
        "macula": {"ru": "макула (центр сетчатки)", "en": "macula (centre of the retina)", "kk": "макула (тордың орталығы)"},
        "vascular_arcades": {"ru": "зона сосудистых аркад", "en": "vascular arcades", "kk": "тамыр аркалары аймағы"},
        "periphery": {"ru": "периферия сетчатки", "en": "retinal periphery", "kk": "тордың шеткі бөлігі"},
    },
    "quadrants": {
        "superior_temporal": {"ru": "верхне-височный сектор", "en": "superotemporal sector", "kk": "жоғарғы самай секторы"},
        "inferior_temporal": {"ru": "нижне-височный сектор", "en": "inferotemporal sector", "kk": "төменгі самай секторы"},
        "superior_nasal": {"ru": "верхне-носовой сектор", "en": "superonasal sector", "kk": "жоғарғы мұрын секторы"},
        "inferior_nasal": {"ru": "нижне-носовой сектор", "en": "inferonasal sector", "kk": "төменгі мұрын секторы"},
    },
    "disease": {
        "dr": {"ru": "Диабетическая ретинопатия", "en": "Diabetic retinopathy", "kk": "Диабеттік ретинопатия"},
        "glaucoma": {"ru": "Глаукома", "en": "Glaucoma", "kk": "Глаукома"},
        "cataract": {"ru": "Катаракта", "en": "Cataract", "kk": "Катаракта"},
    },
    "stage": [
        {"ru": "нет ДР", "en": "no DR", "kk": "ДР жоқ"},
        {"ru": "лёгкая непролиферативная", "en": "mild non-proliferative", "kk": "жеңіл пролиферативті емес"},
        {"ru": "умеренная непролиферативная", "en": "moderate non-proliferative", "kk": "орташа пролиферативті емес"},
        {"ru": "тяжёлая непролиферативная", "en": "severe non-proliferative", "kk": "ауыр пролиферативті емес"},
        {"ru": "пролиферативная", "en": "proliferative", "kk": "пролиферативті"},
    ],
    # what clinicians look for at each ICDR stage — the model was trained on these grades
    "stage_signs": [
        {"ru": "", "en": "", "kk": ""},
        {"ru": "единичные микроаневризмы — самый ранний признак поражения капилляров",
         "en": "isolated microaneurysms — the earliest sign of capillary damage",
         "kk": "жекелеген микроаневризмалар — капилляр зақымының ең ерте белгісі"},
        {"ru": "микроаневризмы, точечные и пятнистые кровоизлияния, твёрдые экссудаты, возможны «ватные» очаги",
         "en": "microaneurysms, dot and blot haemorrhages, hard exudates, possibly cotton-wool spots",
         "kk": "микроаневризмалар, нүктелі және дақты қан құйылулар, қатты экссудаттар, «мақта тәрізді» ошақтар болуы мүмкін"},
        {"ru": "множественные кровоизлияния во всех квадрантах, чёткообразные вены, интраретинальные микрососудистые аномалии (IRMA)",
         "en": "many haemorrhages in all quadrants, venous beading, intraretinal microvascular abnormalities (IRMA)",
         "kk": "барлық квадранттарда көптеген қан құйылулар, веналардың моншақ тәрізді болуы, IRMA"},
        {"ru": "новообразованные сосуды (неоваскуляризация) и/или кровоизлияние в стекловидное тело",
         "en": "new vessels (neovascularisation) and/or vitreous haemorrhage",
         "kk": "жаңа тамырлар (неоваскуляризация) және/немесе шыны тәрізді денеге қан құйылу"},
    ],
}


def tr(d: dict, lang: str) -> str:
    return d.get(lang) or d["ru"]


def _pct(v: float) -> int:
    return int(round(100 * v))


def _regions_text(regions: list, lang: str, top: int = 2) -> str:
    parts = [f"{tr(T['regions'][r['region']], lang)} ({r['share']}%)" for r in regions[:top] if r["share"] >= 10]
    return ", ".join(parts) if parts else tr(T["regions"][regions[0]["region"]], lang)


def disease_finding(key: str, st: dict, stage: int, regions: list, spots: list, macula: bool, quality: dict,
                    focus: dict, an: dict, lang: str) -> dict:
    """Why the model answered what it answered for one disease."""
    status = st["status"]
    p = _pct(st["probability"])
    thr = _pct(st["threshold"])
    name = tr(T["disease"][key], lang)
    where = _regions_text(regions, lang)
    sector = tr(T["quadrants"][quadrant(an, *focus["peak"])], lang)
    ev: list[str] = []

    L = {
        "prob": {"ru": f"Вероятность по оценке ИИ — {p}% (порог решения {thr}%).",
                 "en": f"AI probability {p}% (decision threshold {thr}%).",
                 "kk": f"ЖИ бағалауы бойынша ықтималдық — {p}% (шешім шегі {thr}%)."},
        "where": {"ru": f"Сильнее всего на ответ повлияли: {where}; пик внимания — {sector}.",
                  "en": f"The answer relied most on: {where}; attention peak in the {sector}.",
                  "kk": f"Жауапқа ең көп әсер еткен аймақтар: {where}; назар шыңы — {sector}."},
    }
    ev.append(tr(L["prob"], lang))
    if status != "negative":
        ev.append(tr(L["where"], lang))

    if key == "dr":
        if status != "negative":
            st_name = tr(T["stage"][max(stage, 1)], lang)
            signs = tr(T["stage_signs"][max(stage, 1)], lang)
            ev.append(tr({"ru": f"Картина соответствует стадии «{st_name}». Для неё характерны: {signs}.",
                          "en": f"The picture matches the “{st_name}” stage, typically: {signs}.",
                          "kk": f"Көрініс «{st_name}» сатысына сәйкес. Оған тән: {signs}."}, lang))
            for i, h in enumerate(spots, 1):
                reg = tr(T["regions"][h["region"]], lang)
                sec = tr(T["quadrants"][h["sector"]], lang)
                ev.append(tr({"ru": f"Подозрительный участок {i}: {sec}, {reg}, {h['macula_dd']} диаметра диска от центра макулы (сила сигнала {h['strength']}%).",
                              "en": f"Suspicious area {i}: {sec}, {reg}, {h['macula_dd']} disc diameters from the macular centre (signal {h['strength']}%).",
                              "kk": f"Күдікті аймақ {i}: {sec}, {reg}, макула орталығынан {h['macula_dd']} диск диаметрі (сигнал күші {h['strength']}%)."}, lang))
            if macula:
                ev.append(tr({"ru": "Изменения в пределах 1 диаметра диска от центра макулы — есть риск диабетического макулярного отёка, он угрожает центральному зрению.",
                              "en": "Changes within 1 disc diameter of the macular centre — risk of diabetic macular oedema, which threatens central vision.",
                              "kk": "Өзгерістер макула орталығынан 1 диск диаметрі шегінде — орталық көруге қауіп төндіретін макулярлық ісіну қаупі бар."}, lang))
        else:
            ev.append(tr({"ru": "Микроаневризм, кровоизлияний и экссудатов, характерных для ДР, ИИ не выявил.",
                          "en": "No microaneurysms, haemorrhages or exudates typical of DR were detected.",
                          "kk": "ДР-ға тән микроаневризмалар, қан құйылулар мен экссудаттар анықталмады."}, lang))

    elif key == "glaucoma":
        disc_share = focus["shares"].get("optic_disc", 0)
        if status != "negative":
            ev.append(tr({"ru": f"{disc_share}% внимания ИИ приходится на диск зрительного нерва — главную структуру, по которой врач оценивает глаукому.",
                          "en": f"{disc_share}% of the AI attention is on the optic disc — the key structure a doctor checks for glaucoma.",
                          "kk": f"ЖИ назарының {disc_share}% көру нервінің дискіне түседі — дәрігер глаукоманы бағалайтын негізгі құрылым."}, lang))
            ev.append(tr({"ru": "ИИ обучен на снимках, где глаукому выявляли по увеличенной экскавации (впадине) диска, истончению нейроретинального пояска и асимметрии диска, — картина похожа на эти изменения.",
                          "en": "The AI learned from images where glaucoma was identified by an enlarged optic cup, a thinner neuroretinal rim and disc asymmetry — this picture resembles those changes.",
                          "kk": "ЖИ глаукома диск ойығының үлкеюі, нейроретиналдық белдеудің жұқаруы және диск асимметриясы бойынша анықталған суреттерде оқыды — бұл көрініс сол өзгерістерге ұқсайды."}, lang))
            ev.append(tr({"ru": "Подтвердить или исключить глаукому можно только измерением внутриглазного давления, ОКТ диска и полем зрения.",
                          "en": "Glaucoma can only be confirmed by eye-pressure measurement, optic-disc OCT and a visual-field test.",
                          "kk": "Глаукоманы тек көз ішілік қысымды өлшеу, дискінің ОКТ-сы және көру өрісі арқылы растауға болады."}, lang))
        else:
            ev.append(tr({"ru": "Диск зрительного нерва без подозрительных изменений по оценке ИИ.",
                          "en": "The optic disc shows no suspicious changes according to the AI.",
                          "kk": "ЖИ бағалауы бойынша көру нервінің дискінде күдікті өзгерістер жоқ."}, lang))

    elif key == "cataract":
        vc = quality["vessel_contrast"]
        hazy = vc < VESSEL_LOW * 1.6
        if status != "negative":
            ev.append(tr({"ru": f"Снимок выглядит «затуманенным»: контраст сосудов {vc} (у чётких снимков обычно выше 1,5). Так помутневший хрусталик рассеивает свет.",
                          "en": f"The image looks hazy: vessel contrast {vc} (clear images are usually above 1.5). This is how a cloudy lens scatters light.",
                          "kk": f"Сурет «бұлыңғыр»: тамырлар контрасы {vc} (анық суреттерде әдетте 1,5-тен жоғары). Бұлыңғыр көз бұршағы жарықты осылай шашыратады."}, lang))
            ev.append(tr({"ru": "Внимание ИИ распределено по всему снимку, а не по отдельным очагам — это типично для катаракты.",
                          "en": "AI attention is spread over the whole image rather than on single spots — typical of cataract.",
                          "kk": "ЖИ назары жеке ошақтарға емес, бүкіл суретке таралған — бұл катарактаға тән."}, lang))
        elif hazy:
            ev.append(tr({"ru": "Снимок немного размыт, но характерного для катаракты помутнения ИИ не нашёл.",
                          "en": "The image is slightly blurred, but the AI found no haze typical of cataract.",
                          "kk": "Сурет сәл бұлыңғыр, бірақ ЖИ катарактаға тән бұлыңғырлықты таппады."}, lang))
        else:
            ev.append(tr({"ru": "Сосуды и диск видны чётко — признаков помутнения хрусталика нет.",
                          "en": "Vessels and disc are sharp — no sign of lens opacity.",
                          "kk": "Тамырлар мен диск анық көрінеді — көз бұршағының бұлыңғырлығы белгілері жоқ."}, lang))

    if status == "uncertain":
        ev.append(tr({"ru": "Вероятность близка к порогу — ИИ не делает вывод сам и передаёт решение врачу.",
                      "en": "The probability is close to the threshold — the AI does not decide on its own and refers to a doctor.",
                      "kk": "Ықтималдық шекке жақын — ЖИ өзі шешім қабылдамай, дәрігерге жібереді."}, lang))

    head = {"positive": {"ru": "признаки обнаружены", "en": "signs detected", "kk": "белгілер анықталды"},
            "uncertain": {"ru": "сомнительный результат", "en": "inconclusive", "kk": "күмәнді нәтиже"},
            "negative": {"ru": "признаков нет", "en": "no signs", "kk": "белгілер жоқ"}}[status]
    return {"disease": key, "name": name, "status": status, "title": f"{name}: {tr(head, lang)}", "evidence": ev}


# ============================================================================ recommendations
URGENCY_ORDER = ["routine", "planned", "soon", "urgent", "emergency"]


def _rec(urgency, timeframe, headline, actions, exams=None):
    return dict(urgency=urgency, timeframe=timeframe, headline=headline, actions=actions, exams=exams or [])


def disease_recommendation(key: str, status: str, stage: int, macula: bool, lang: str) -> dict | None:
    L = lambda d: tr(d, lang)  # noqa: E731
    if status == "negative":
        return None
    if key == "dr":
        if status == "uncertain":
            return _rec("soon", L({"ru": "в течение 2–4 недель", "en": "within 2–4 weeks", "kk": "2–4 апта ішінде"}),
                        L({"ru": "Осмотр глазного дна у офтальмолога для уточнения", "en": "Fundus examination by an ophthalmologist to clarify",
                           "kk": "Нақтылау үшін офтальмологта көз түбін тексеру"}),
                        [L({"ru": "Сдать анализ на гликированный гемоглобин (HbA1c) и глюкозу крови", "en": "Check HbA1c and blood glucose",
                            "kk": "Гликирленген гемоглобин (HbA1c) мен қандағы глюкозаны тексеру"})],
                        [L({"ru": "Осмотр глазного дна с расширением зрачка", "en": "Dilated fundus examination", "kk": "Қарашықты кеңейтіп көз түбін тексеру"})])
        if stage >= 4:
            return _rec("urgent", L({"ru": "срочно, в течение 24–72 часов", "en": "urgently, within 24–72 hours", "kk": "шұғыл, 24–72 сағат ішінде"}),
                        L({"ru": "Срочно посетите офтальмолога (ретинолога): подозрение на пролиферативную ДР",
                           "en": "See an ophthalmologist (retina specialist) urgently: suspected proliferative DR",
                           "kk": "Офтальмологқа (ретинологқа) шұғыл барыңыз: пролиферативті ДР күдігі"}),
                        [L({"ru": "Не откладывайте визит: новые сосуды могут кровоточить и вызвать резкую потерю зрения",
                            "en": "Do not postpone: new vessels can bleed and cause sudden vision loss",
                            "kk": "Сапарды кейінге қалдырмаңыз: жаңа тамырлар қансырап, көруді кенет жоғалтуы мүмкін"}),
                         L({"ru": "Избегайте подъёма тяжестей и резких наклонов до осмотра", "en": "Avoid heavy lifting and straining until examined",
                            "kk": "Тексерілгенге дейін ауыр көтеруден және кенет еңкеюден аулақ болыңыз"}),
                         L({"ru": "Сообщите эндокринологу: нужен строгий контроль сахара и давления", "en": "Inform your endocrinologist: strict glucose and blood-pressure control is needed",
                            "kk": "Эндокринологқа хабарлаңыз: қант пен қысымды қатаң бақылау қажет"})],
                        [L({"ru": "Осмотр глазного дна с расширением зрачка", "en": "Dilated fundus examination", "kk": "Қарашықты кеңейтіп көз түбін тексеру"}),
                         L({"ru": "ОКТ сетчатки", "en": "Retinal OCT", "kk": "Тордың ОКТ-сы"}),
                         L({"ru": "Решение о лазерной коагуляции или анти-VEGF терапии", "en": "Decision on laser photocoagulation or anti-VEGF therapy",
                            "kk": "Лазерлік коагуляция немесе анти-VEGF терапиясы туралы шешім"})])
        if stage == 3:
            return _rec("urgent", L({"ru": "в течение 7 дней", "en": "within 7 days", "kk": "7 күн ішінде"}),
                        L({"ru": "Срочная консультация ретинолога: тяжёлая непролиферативная ДР, высокий риск перехода в пролиферативную",
                           "en": "Urgent retina consultation: severe non-proliferative DR with a high risk of progression",
                           "kk": "Ретинологтың шұғыл кеңесі: ауыр пролиферативті емес ДР, асқыну қаупі жоғары"}),
                        [L({"ru": "Строгий контроль глюкозы (цель HbA1c — по согласованию с эндокринологом, обычно < 7%)",
                            "en": "Tight glucose control (HbA1c target agreed with your endocrinologist, usually < 7%)",
                            "kk": "Глюкозаны қатаң бақылау (HbA1c мақсаты эндокринологпен келісіледі, әдетте < 7%)"}),
                         L({"ru": "Контроль артериального давления (обычно < 130/80) и холестерина", "en": "Control blood pressure (usually < 130/80) and cholesterol",
                            "kk": "Артериялық қысымды (әдетте < 130/80) және холестеринді бақылау"})],
                        [L({"ru": "Осмотр глазного дна с расширением зрачка", "en": "Dilated fundus examination", "kk": "Қарашықты кеңейтіп көз түбін тексеру"}),
                         L({"ru": "ОКТ сетчатки", "en": "Retinal OCT", "kk": "Тордың ОКТ-сы"}),
                         L({"ru": "Возможна панретинальная лазеркоагуляция", "en": "Panretinal laser may be considered", "kk": "Панретиналды лазерлік коагуляция қарастырылуы мүмкін"})])
        if stage == 2:
            r = _rec("soon", L({"ru": "в течение 1 месяца", "en": "within 1 month", "kk": "1 ай ішінде"}),
                     L({"ru": "Консультация офтальмолога: умеренная непролиферативная ДР", "en": "Ophthalmology consultation: moderate non-proliferative DR",
                        "kk": "Офтальмолог кеңесі: орташа пролиферативті емес ДР"}),
                     [L({"ru": "Контроль сахара, давления и холестерина вместе с эндокринологом/терапевтом",
                         "en": "Control glucose, blood pressure and cholesterol with your endocrinologist/GP",
                         "kk": "Эндокринологпен/терапевтпен бірге қант, қысым және холестеринді бақылау"}),
                      L({"ru": "Повторный осмотр сетчатки каждые 3–6 месяцев", "en": "Repeat retinal check every 3–6 months",
                         "kk": "Торды әр 3–6 ай сайын қайта тексеру"})],
                     [L({"ru": "Осмотр глазного дна с расширением зрачка", "en": "Dilated fundus examination", "kk": "Қарашықты кеңейтіп көз түбін тексеру"}),
                      L({"ru": "ОКТ макулы", "en": "Macular OCT", "kk": "Макуланың ОКТ-сы"})])
        else:
            r = _rec("planned", L({"ru": "в течение 3 месяцев", "en": "within 3 months", "kk": "3 ай ішінде"}),
                     L({"ru": "Плановый визит к офтальмологу: ранние признаки ДР", "en": "Planned ophthalmology visit: early signs of DR",
                        "kk": "Офтальмологқа жоспарлы бару: ДР-ның ерте белгілері"}),
                     [L({"ru": "Держите сахар в целевом диапазоне — на этой стадии изменения обратимы", "en": "Keep glucose in the target range — changes at this stage can regress",
                         "kk": "Қантты мақсатты деңгейде ұстаңыз — бұл сатыда өзгерістер қайтымды"}),
                      L({"ru": "Повторный скрининг через 6–12 месяцев", "en": "Repeat screening in 6–12 months", "kk": "6–12 айдан кейін қайта скрининг"})],
                     [L({"ru": "Осмотр глазного дна с расширением зрачка", "en": "Dilated fundus examination", "kk": "Қарашықты кеңейтіп көз түбін тексеру"})])
        if macula:
            r["urgency"] = URGENCY_ORDER[max(URGENCY_ORDER.index(r["urgency"]), URGENCY_ORDER.index("urgent"))]
            r["timeframe"] = L({"ru": "в течение 7 дней", "en": "within 7 days", "kk": "7 күн ішінде"})
            plain_oct = L({"ru": "ОКТ макулы", "en": "Macular OCT", "kk": "Макуланың ОКТ-сы"})
            r["exams"] = [e for e in r["exams"] if e != plain_oct]
            r["exams"].append(L({"ru": "ОКТ макулы — исключить макулярный отёк", "en": "Macular OCT to rule out macular oedema",
                                 "kk": "Макулярлық ісінуді жоққа шығару үшін макуланың ОКТ-сы"}))
        return r
    if key == "glaucoma":
        sure = status == "positive"
        return _rec("soon" if sure else "planned",
                    L({"ru": "в течение 2 недель", "en": "within 2 weeks", "kk": "2 апта ішінде"}) if sure else
                    L({"ru": "в течение 1 месяца", "en": "within 1 month", "kk": "1 ай ішінде"}),
                    L({"ru": "Обследование у офтальмолога на глаукому", "en": "Glaucoma work-up by an ophthalmologist", "kk": "Офтальмологта глаукомаға тексерілу"}),
                    [L({"ru": "Глаукома долго протекает без симптомов, а потерянное зрение не восстанавливается — не откладывайте",
                        "en": "Glaucoma is silent for years and lost vision does not return — do not delay",
                        "kk": "Глаукома ұзақ уақыт білінбейді, ал жоғалған көру қалпына келмейді — кейінге қалдырмаңыз"}),
                     L({"ru": "Сообщите врачу, если глаукома есть у родственников", "en": "Tell the doctor if relatives have glaucoma",
                        "kk": "Туыстарыңызда глаукома болса, дәрігерге айтыңыз"})],
                    [L({"ru": "Измерение внутриглазного давления (тонометрия)", "en": "Eye-pressure measurement (tonometry)", "kk": "Көз ішілік қысымды өлшеу (тонометрия)"}),
                     L({"ru": "ОКТ диска зрительного нерва", "en": "Optic-disc OCT", "kk": "Көру нерві дискінің ОКТ-сы"}),
                     L({"ru": "Компьютерная периметрия (поле зрения)", "en": "Visual-field test (perimetry)", "kk": "Компьютерлік периметрия (көру өрісі)"})])
    if key == "cataract":
        return _rec("planned", L({"ru": "в течение 1–3 месяцев", "en": "within 1–3 months", "kk": "1–3 ай ішінде"}),
                    L({"ru": "Плановая консультация офтальмолога по поводу катаракты", "en": "Planned ophthalmology consultation for cataract",
                       "kk": "Катаракта бойынша офтальмологтың жоспарлы кеңесі"}),
                    [L({"ru": "Если зрение мешает читать, водить машину или работать — обсудите операцию (факоэмульсификацию)",
                        "en": "If vision interferes with reading, driving or work — discuss surgery (phacoemulsification)",
                        "kk": "Көру оқуға, көлік жүргізуге немесе жұмысқа кедергі келтірсе — операцияны (факоэмульсификация) талқылаңыз"}),
                     L({"ru": "Катаракта мешает видеть сетчатку — после лечения нужен повторный скрининг глазного дна",
                        "en": "Cataract hides the retina — repeat fundus screening after treatment",
                        "kk": "Катаракта торды жасырады — емнен кейін көз түбін қайта скринингтен өткізу қажет"})],
                    [L({"ru": "Биомикроскопия хрусталика (щелевая лампа)", "en": "Slit-lamp examination of the lens", "kk": "Көз бұршағының биомикроскопиясы (саңылаулы шам)"}),
                     L({"ru": "Проверка остроты зрения", "en": "Visual acuity test", "kk": "Көру өткірлігін тексеру"})])
    return None


RED_FLAGS = [
    {"ru": "внезапное снижение или потеря зрения", "en": "sudden blurring or loss of vision", "kk": "көрудің кенет нашарлауы немесе жоғалуы"},
    {"ru": "«занавеска» или тень перед глазом", "en": "a curtain or shadow over your vision", "kk": "көз алдында «перде» немесе көлеңке"},
    {"ru": "вспышки света и много новых «плавающих мушек»", "en": "flashes of light and many new floaters", "kk": "жарық жарқылдары және көптеген жаңа «ұшатын шыбындар»"},
    {"ru": "сильная боль в глазу с покраснением, радужные круги вокруг света, головная боль с тошнотой",
     "en": "severe eye pain with redness, halos around lights, headache with nausea",
     "kk": "көздің қатты ауыруы мен қызаруы, жарық айналасындағы кемпірқосақ шеңберлер, жүрек айнитын бас ауруы"},
]


def recommendations(screening: dict, stage: int, macula: bool, quality: dict, lang: str) -> dict:
    L = lambda d: tr(d, lang)  # noqa: E731
    per = {}
    for k in DISEASES:
        r = disease_recommendation(k, screening[k]["status"], stage, macula, lang)
        if r:
            per[k] = r
    if per:
        top = max(per.values(), key=lambda r: URGENCY_ORDER.index(r["urgency"]))
        urgency, timeframe, headline = top["urgency"], top["timeframe"], top["headline"]
    else:
        urgency = "routine"
        timeframe = L({"ru": "через 12 месяцев", "en": "in 12 months", "kk": "12 айдан кейін"})
        headline = L({"ru": "Повторный скрининг через год", "en": "Repeat screening in one year", "kk": "Бір жылдан кейін қайта скрининг"})

    general = []
    if per:
        general.append(L({"ru": "Возьмите с собой этот отчёт и список принимаемых лекарств", "en": "Bring this report and your medication list to the visit",
                          "kk": "Осы есепті және қабылдайтын дәрілер тізімін ала барыңыз"}))
    general.append(L({"ru": "При диабете — осмотр глазного дна не реже 1 раза в год, даже если зрение хорошее",
                      "en": "With diabetes — a fundus check at least once a year, even if vision is fine",
                      "kk": "Диабет болса — көру жақсы болса да, көз түбін жылына кемінде 1 рет тексеру"}))
    if quality["issues"]:
        general.insert(0, L({"ru": "Качество снимка снижено — по возможности переснимите глаз (без бликов, в фокусе, диск и макула в кадре)",
                             "en": "Image quality is reduced — retake the photo if possible (in focus, no glare, disc and macula in frame)",
                             "kk": "Сурет сапасы төмен — мүмкін болса, қайта түсіріңіз (фокуста, жарқылсыз, диск пен макула кадрда)"}))
    labels = {"routine": {"ru": "Плановый контроль", "en": "Routine follow-up", "kk": "Жоспарлы бақылау"},
              "planned": {"ru": "Плановый визит", "en": "Planned visit", "kk": "Жоспарлы бару"},
              "soon": {"ru": "Визит в ближайшее время", "en": "Visit soon", "kk": "Жақын арада бару"},
              "urgent": {"ru": "Срочно", "en": "Urgent", "kk": "Шұғыл"},
              "emergency": {"ru": "Неотложно", "en": "Emergency", "kk": "Кезек күттірмейтін"}}
    return dict(
        urgency=urgency, urgency_label=L(labels[urgency]), timeframe=timeframe, headline=headline,
        by_disease=per, general=general,
        red_flags_title=L({"ru": "Немедленно обратитесь за неотложной помощью, если появились:",
                           "en": "Seek emergency care immediately if you notice:",
                           "kk": "Мына белгілер пайда болса, дереу шұғыл көмекке жүгініңіз:"}),
        red_flags=[L(f) for f in RED_FLAGS],
    )


# ============================================================================ heatmap image
def _colormap(v: np.ndarray) -> np.ndarray:
    """Dark-purple → orange → yellow, like the presentation slides."""
    stops = np.array([[0.0, 0, 0, 0], [0.35, 90, 20, 120], [0.6, 220, 60, 60], [0.8, 255, 160, 40], [1.0, 255, 240, 160]])
    out = np.zeros(v.shape + (3,), dtype=np.float32)
    for c in range(3):
        out[..., c] = np.interp(v, stops[:, 0], stops[:, c + 1])
    return out


def heatmap_image(rgb: np.ndarray, att: np.ndarray, mask: np.ndarray, size: int = 384, strength: float = 1.0) -> str:
    """strength < 1 dims the map when the model sees little (a healthy eye should look almost dark)."""
    a = np.clip((att - 0.15) / 0.85, 0, 1) * strength
    alpha = (a ** 0.8 * 0.75)[..., None]
    over = rgb * (1 - alpha) + _colormap(a) * alpha
    over = np.where(mask[..., None], over, 0)
    img = Image.fromarray(np.clip(over, 0, 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# ============================================================================ entry point
def explain(rgb: np.ndarray, outputs: dict, screening: dict, cam_head: CamHead, lang: str = "ru",
            with_images: bool = True) -> dict:
    lang = lang if lang in LANGS else "ru"
    s = rgb.shape[0]
    an = anatomy(rgb)
    regions_m = region_masks(an, s)
    stage = int(screening["dr_stage"]["class"])

    cams, focus = {}, {}
    for k in DISEASES:
        cls = None
        cam = cam_head.cam(outputs["feat"], outputs["feat_flip"], k, cls)
        att = upsample(cam, s) * an["mask"]
        cams[k] = att
        tot = float(att.sum()) or 1.0
        shares = {r: int(round(100 * float(att[m].sum()) / tot)) for r, m in regions_m.items()}
        peak = np.unravel_index(int(np.argmax(att)), att.shape)
        focus[k] = dict(shares=shares, peak=(float(peak[0]), float(peak[1])))

    quality = image_quality(rgb, an)
    spots = {k: hotspots(cams[k], an) for k in DISEASES}
    # ICDR referable maculopathy: changes within 1 disc diameter of the fovea
    macula = screening["dr"]["status"] != "negative" and any(h["macula_dd"] <= 1.0 for h in spots["dr"])

    findings = []
    for k in DISEASES:
        regs = sorted(({"region": r, "share": v} for r, v in focus[k]["shares"].items()), key=lambda r: -r["share"])
        f = disease_finding(k, screening[k], stage, regs, spots[k], macula, quality, focus[k], an, lang)
        f["attention"] = regs
        f["hotspots"] = spots[k] if screening[k]["status"] != "negative" else []
        findings.append(f)
    order = {"positive": 0, "uncertain": 1, "negative": 2}
    findings.sort(key=lambda f: order[f["status"]])

    flagged = [f for f in findings if f["status"] != "negative"]
    if flagged:
        summary = tr({"ru": "ИИ обосновывает ответ так: ", "en": "How the AI reached its answer: ", "kk": "ЖИ жауабын былай негіздейді: "}, lang) + \
            "; ".join(f["title"] for f in flagged) + "."
    else:
        summary = tr({"ru": "Ни по одной из трёх болезней ИИ не нашёл значимых изменений: сосуды, диск зрительного нерва и макула выглядят типично для здорового глаза.",
                      "en": "The AI found no significant changes for any of the three diseases: vessels, optic disc and macula look typical of a healthy eye.",
                      "kk": "ЖИ үш аурудың ешқайсысы бойынша елеулі өзгерістер таппады: тамырлар, көру нервінің дискі және макула сау көзге тән."}, lang)

    qtext = []
    Q = {"too_dark": {"ru": "снимок тёмный", "en": "image is too dark", "kk": "сурет тым қараңғы"},
         "overexposed": {"ru": "снимок пересвечен", "en": "image is overexposed", "kk": "сурет тым жарық"},
         "small_field": {"ru": "в кадре мало сетчатки", "en": "little retina in frame", "kk": "кадрда тор аз"},
         "blurred": {"ru": "снимок размыт или затуманен", "en": "image is blurred or hazy", "kk": "сурет бұлыңғыр"}}
    for i in quality["issues"]:
        qtext.append(tr(Q[i], lang))

    main = flagged[0]["disease"] if flagged else "dr"
    out = dict(
        method="Grad-CAM",
        summary=summary,
        findings=findings,
        anatomy=dict(optic_disc={"y": round(an["disc"][0] / s, 3), "x": round(an["disc"][1] / s, 3)},
                     macula={"y": round(an["macula"][0] / s, 3), "x": round(an["macula"][1] / s, 3)}),
        macula_involved=macula,
        image_quality={**quality, "issues_text": qtext},
        heatmap_disease=main,
    )
    if with_images:
        strength = {k: float(np.clip(screening[k]["probability"] / max(screening[k]["threshold"], 1e-3), 0.2, 1.0))
                    for k in DISEASES}
        out["heatmaps"] = {k: heatmap_image(rgb, cams[k], an["mask"], strength=strength[k]) for k in DISEASES
                           if k == main or screening[k]["status"] != "negative"}
        out["heatmap"] = out["heatmaps"][main]
    rec = recommendations(screening, stage, macula, quality, lang)
    return out, rec
