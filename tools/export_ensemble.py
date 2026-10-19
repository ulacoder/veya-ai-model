"""Export the whole VEYA v2 ensemble for the API: three base models + the MLP "brain".

    pip install -r tools/requirements-ensemble.txt
    python tools/export_ensemble.py --src path/to/unzipped_checkpoints

--src is searched recursively for every `best.pt` written by training/veya_v2/train.py (the model key is
read from the checkpoint itself) and for `brain.pt` from training/veya_v2/brain.py (plus `report.json`
next to it, if present). Output goes to model_bundle/ensemble/:

    <key>.onnx      graph (+ <key>.weights, memory-mapped by ONNX Runtime); image 0..255 → TTA-averaged logits, probabilities and the pooled embedding
                    (EfficientNetV2-S also returns its feature maps for Grad-CAM)
    brain.npz       brain MLPs with BatchNorm folded in, PCA of the embeddings, feature scaling
    ensemble.json   model list, input sizes, thresholds and doubt zones, test metrics

ConvNeXt and ViT are int8-quantized (dynamic, MatMul weights) so the three models fit a 512 MB instance;
the script prints how far the quantized outputs drift from PyTorch. --no-quantize keeps everything fp32.
The final answer is ALWAYS the brain's (its thresholds come from brain.pt).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from training.veya_v2 import train as T  # noqa: E402

OUT = ROOT / "model_bundle" / "ensemble"
ORDER = ["effv2s", "convnext", "vit"]
CAM_MODEL = "effv2s"
QUANTIZE = {"convnext", "vit"}


class Wrapper(nn.Module):
    """Normalisation + flip-TTA; returns logits (for the brain), probabilities and the embedding."""

    def __init__(self, model: T.VeyaNet, with_feats: bool, batch_tta: bool):
        super().__init__()
        self.m = model
        self.with_feats = with_feats
        self.batch_tta = batch_tta
        self.register_buffer("mean", torch.tensor(T.MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(T.STD).view(1, 3, 1, 1))

    def _pass(self, x):
        fmap = self.m.backbone.forward_features(x)
        f = self.m.backbone.forward_head(fmap, pre_logits=True)
        h = self.m.neck(f)
        g = self.m.heads["dr_grade"](h)
        b = torch.cat([self.m.heads[k](h) for k in T.BINARY], 1)
        return g, b, f, fmap

    def forward(self, image):
        x = (image - self.mean) / self.std
        flipped = torch.flip(x, dims=[3])
        if self.batch_tta:
            # one batch of two: each weight feeds one node (the int8 quantizer breaks on shared Gemm weights)
            g, b, f, fmap = self._pass(torch.cat([x, flipped]))
            g, b, emb, m1, m2 = g.mean(0, keepdim=True), b.mean(0, keepdim=True), f.mean(0, keepdim=True), fmap[0:1], fmap[1:2]
        else:
            # two passes one after another: half the activation memory of a batch (matters at 512 px)
            g1, b1, f1, m1 = self._pass(x)
            g2, b2, f2, m2 = self._pass(flipped)
            g, b, emb = (g1 + g2) / 2, (b1 + b2) / 2, (f1 + f2) / 2
        p = torch.sigmoid(b)
        out = (g, b, emb, torch.softmax(g, 1), p[:, 0], p[:, 1], p[:, 2])
        return out + ((m1, m2) if self.with_feats else ())


NAMES = ["grade_logits", "bin_logits", "emb", "dr_grade", "dr", "glaucoma", "cataract"]


def synthetic_fundus(size: int) -> np.ndarray:
    """Orange disc with a bright optic disc and a dark macula on black, [1, 3, S, S] in 0..255."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    r = np.hypot(yy - 0.5, xx - 0.5)
    disc = np.exp(-((yy - 0.5) ** 2 + (xx - 0.3) ** 2) / 0.002)
    mac = np.exp(-((yy - 0.52) ** 2 + (xx - 0.62) ** 2) / 0.004)
    base = np.clip(1 - r / 0.48, 0, 1) ** 0.3 * (r < 0.48)
    img = np.stack([200 * base + 50 * disc - 40 * mac, 90 * base + 120 * disc - 30 * mac, 40 * base + 80 * disc], 0)
    return np.clip(img, 0, 255)[None].astype(np.float32)


def find_checkpoints(src: Path) -> tuple[dict, Path]:
    ckpts = {}
    for p in sorted(src.rglob("best.pt")):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        key = ck.get("key") or p.parent.name
        if key in T.MODELS:
            ckpts[key] = (p, ck)
            print(f"  {key:9s} ← {p}  (val score {ck.get('metrics', {}).get('score', float('nan')):.4f})")
    brains = sorted(src.rglob("brain.pt"))
    if not brains:
        sys.exit("brain.pt not found under --src")
    return ckpts, brains[0]


def export_model(key: str, ck: dict, quantize: bool) -> dict:
    size = int(ck.get("cfg", {}).get("size", T.MODELS[key]["size"]))
    model = T.build_model(key, pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval()
    with_feats = key == CAM_MODEL
    wrapper = Wrapper(model, with_feats, batch_tta=quantize).eval()
    names = NAMES + (["feat", "feat_flip"] if with_feats else [])
    dummy = torch.rand(1, 3, size, size) * 255
    fp32 = OUT / f"{key}.fp32.onnx"
    torch.onnx.export(wrapper, dummy, str(fp32), input_names=["image"], output_names=names,
                      opset_version=17, dynamo=False)
    final = OUT / f"{key}.onnx"
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        import onnx
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = OUT / f"{key}.pre.onnx"
        quant_pre_process(str(fp32), str(pre), skip_optimization=True, skip_symbolic_shape=True)
        # backbone MatMuls only: the small neck/heads stay fp32 so the brain's inputs barely move
        graph = onnx.load(str(pre), load_external_data=False).graph
        keep = [n.name for n in graph.node if "/neck" in n.name or "/heads" in n.name]
        keep += [f"{name}_MatMul" for name in keep]  # the quantizer renames Gemm → MatMul
        quantize_dynamic(str(pre), str(final), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"],
                         nodes_to_exclude=keep, extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
        fp32.unlink()
        pre.unlink()
    else:
        fp32.replace(final)

    # drift check against PyTorch on a fundus-like image (pure noise is not a fair input for a trained net)
    probe = torch.from_numpy(synthetic_fundus(size))
    with torch.no_grad():
        ref = [t.numpy() for t in wrapper(probe)]
    sess = ort.InferenceSession(str(final), providers=["CPUExecutionProvider"])
    got = sess.run(names, {"image": probe.numpy()})
    probs = max(float(np.abs(ref[i] - got[i]).max()) for i in range(3, 7))
    emb_corr = float(np.abs(ref[2] - got[2]).max() / (np.abs(ref[2]).max() + 1e-9))
    print(f"  {key:9s} {size}px  {final.stat().st_size / 2**20:6.1f} MB  "
          f"max |Δprob| {probs:.4f}  max rel Δemb {emb_corr:.4f}{'  (int8)' if quantize else ''}")

    # weights in a side file: ORT memory-maps it, so the pages are file-backed and reclaimable instead of
    # anonymous memory (peak anonymous RAM of the three models drops from ~420 to ~280 MB)
    import onnx
    del sess
    proto = onnx.load(str(final))
    (OUT / f"{key}.weights").unlink(missing_ok=True)
    onnx.save_model(proto, str(final), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=f"{key}.weights", size_threshold=1024)
    (OUT / f"{key}.weights").chmod(0o644)

    if with_feats:  # Grad-CAM head weights (same layout as tools/add_explain_outputs.py)
        sd = {k: v.numpy().astype(np.float32) for k, v in model.state_dict().items()}
        np.savez_compressed(OUT / "explain_head.npz", **{
            k.replace(".", "_"): sd[k] for k in sd if k.startswith(("neck.1", "neck.4", "heads."))})
    return {"key": key, "file": final.name, "size": size, "timm": T.MODELS[key]["timm"]}


def effv2s_from_bundle() -> dict:
    """No effv2s best.pt? Reuse the deployed model_bundle/model.onnx (same network, already exported with
    feature maps by tools/add_explain_outputs.py) and expose the logits and the embedding the brain needs."""
    import onnx
    from onnx import helper
    src = ROOT / "model_bundle" / "model.onnx"
    proto = onnx.load(str(src))
    g = proto.graph
    produced = {o for n in g.node for o in n.output}
    pooled = ["/model/backbone/global_pool/flatten/Flatten_output_0", "/model/backbone/global_pool/flatten_1/Flatten_output_0"]
    grade, bins = "/Div_1_output_0", ["/Div_2_output_0", "/Div_3_output_0", "/Div_4_output_0"]
    assert all(t in produced for t in pooled + [grade] + bins), "model.onnx layout differs from the VEYA v2 export"
    one = helper.make_tensor("ens_axis1", onnx.TensorProto.INT64, [1], [1])
    half = helper.make_tensor("ens_half", onnx.TensorProto.FLOAT, [], [0.5])
    g.initializer.extend([one, half])
    nodes = [helper.make_node("Identity", [grade], ["grade_logits"])]
    for i, t in enumerate(bins):
        nodes.append(helper.make_node("Unsqueeze", [t, "ens_axis1"], [f"ens_bin{i}"]))
    nodes += [helper.make_node("Concat", [f"ens_bin{i}" for i in range(3)], ["bin_logits"], axis=1),
              helper.make_node("Add", pooled, ["ens_pool_sum"]),
              helper.make_node("Mul", ["ens_pool_sum", "ens_half"], ["emb"])]
    g.node.extend(nodes)
    for name, shape in (("grade_logits", [1, 5]), ("bin_logits", [1, 3]), ("emb", [1, "emb_dim"])):
        g.output.append(helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape))
    onnx.checker.check_model(proto)
    final = OUT / "effv2s.onnx"
    (OUT / "effv2s.weights").unlink(missing_ok=True)
    onnx.save_model(proto, str(final), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="effv2s.weights", size_threshold=1024)
    (OUT / "effv2s.weights").chmod(0o644)
    import shutil
    shutil.copy(ROOT / "model_bundle" / "explain_head.npz", OUT / "explain_head.npz")
    cfg = json.loads((ROOT / "model_bundle" / "model_config.json").read_text())
    print(f"  effv2s    {cfg['image_size']}px  reused model_bundle/model.onnx (no best.pt found)")
    return {"key": "effv2s", "file": final.name, "size": int(cfg["image_size"]), "timm": T.MODELS["effv2s"]["timm"]}


def export_brain(path: Path, models: list[str]) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    assert list(ck["models"]) == models, f"brain was trained on {ck['models']}, checkpoints found for {models}"
    arrays = {"mu": np.asarray(ck["mu"], np.float32), "sd": np.asarray(ck["sd"], np.float32)}
    for m in models:
        comps, mean = ck["pca"][m]
        arrays[f"pca_{m}_components"] = np.asarray(comps, np.float32)
        arrays[f"pca_{m}_mean"] = np.asarray(mean, np.float32)
    for i, sd in enumerate(ck["nets"]):
        sd = {k: v.numpy().astype(np.float64) for k, v in sd.items()}
        for layer, (lin, bn) in enumerate(((0, 1), (4, 5), (8, 9))):
            w, b = sd[f"body.{lin}.weight"], sd[f"body.{lin}.bias"]
            scale = sd[f"body.{bn}.weight"] / np.sqrt(sd[f"body.{bn}.running_var"] + 1e-5)
            arrays[f"net{i}_w{layer}"] = (w * scale[:, None]).astype(np.float32)
            arrays[f"net{i}_b{layer}"] = ((b - sd[f"body.{bn}.running_mean"]) * scale + sd[f"body.{bn}.bias"]).astype(np.float32)
        for head in ("grade", "bin"):
            arrays[f"net{i}_{head}_w"] = sd[f"{head}.weight"].astype(np.float32)
            arrays[f"net{i}_{head}_b"] = sd[f"{head}.bias"].astype(np.float32)
    np.savez_compressed(OUT / "brain.npz", n_nets=np.array(len(ck["nets"])), **arrays)
    not_brain = [k for k, v in ck.get("use_brain", {}).items() if not v]
    if not_brain:
        print(f"  ! thresholds for {not_brain} were tuned on the plain average; the API still lets the brain decide. "
              f"Re-run training/veya_v2/brain.run_brain(always_brain=True) to retune them.")
    return {"thresholds": {k: float(v) for k, v in ck["thresholds"].items()},
            "doubt_zones": {k: [float(a), float(b)] for k, (a, b) in ck["zones"].items()},
            "n_nets": len(ck["nets"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print("Checkpoints:")
    ckpts, brain_path = find_checkpoints(args.src)
    missing = [k for k in ORDER if k not in ckpts and k != CAM_MODEL]
    if missing:
        sys.exit(f"missing best.pt for {missing}")
    print("Export:")
    models = [export_model(k, ckpts[k][1], quantize=k in QUANTIZE and not args.no_quantize) if k in ckpts
              else effv2s_from_bundle() for k in ORDER]
    print(f"Brain ← {brain_path}")
    brain = export_brain(brain_path, ORDER)
    report = brain_path.parent / "report.json"
    metrics = {}
    if report.exists():
        rep = json.loads(report.read_text())
        metrics = {sp: rep[sp] for sp in ("test", "test_ext") if sp in rep}
    config = {
        "format": "veya-ensemble",
        "model_name": "veya-v2-brain-ensemble",
        "decision": "brain",
        "models": models,
        "cam_model": CAM_MODEL,
        "image_size": max(m["size"] for m in models),
        "dr_grade_names": ["нет ДР", "лёгкая", "умеренная", "тяжёлая", "пролиферативная"],
        "brain": {"file": "brain.npz", "n_nets": brain["n_nets"]},
        "thresholds": brain["thresholds"],
        "doubt_zones": brain["doubt_zones"],
        "metrics": metrics,
    }
    (OUT / "ensemble.json").write_text(json.dumps(config, ensure_ascii=False, indent=1))
    print(f"Done → {OUT}")


if __name__ == "__main__":
    main()
