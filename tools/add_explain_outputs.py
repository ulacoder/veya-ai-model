"""Prepare the VEYA v2 bundle for explained answers (Grad-CAM without PyTorch).

Adds the last backbone feature maps of both TTA passes (original + horizontally flipped) as extra
ONNX outputs `feat` / `feat_flip`, and saves the neck + head weights to `explain_head.npz`.
main.py then computes Grad-CAM in NumPy: the head is a small MLP over the pooled features, so its
gradient is written out analytically.

Run once after every export:  pip install onnx && python tools/add_explain_outputs.py
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

BUNDLE = Path(__file__).resolve().parent.parent / "model_bundle"
FEATS = {"feat": "/model/backbone/bn2/act/Mul_output_0",
         "feat_flip": "/model/backbone/bn2/act_1/Mul_output_0"}
WEIGHTS = ["model.neck.1.weight", "model.neck.1.bias", "model.neck.4.weight", "model.neck.4.bias",
           *[f"model.heads.{h}.{p}" for h in ("dr_grade", "dr", "glaucoma", "cataract") for p in ("weight", "bias")]]

model = onnx.load(str(BUNDLE / "model.onnx"))
graph = model.graph
produced = {o for n in graph.node for o in n.output}
existing = {o.name for o in graph.output}
for alias, tensor in FEATS.items():
    assert tensor in produced, f"tensor {tensor} not found — was the model exported with a different backbone?"
    if alias in existing:
        continue
    graph.node.append(helper.make_node("Identity", [tensor], [alias]))
    graph.output.append(helper.make_tensor_value_info(alias, onnx.TensorProto.FLOAT, [1, "channels", "height", "width"]))

init = {i.name: numpy_helper.to_array(i) for i in graph.initializer if i.name in WEIGHTS}
missing = set(WEIGHTS) - set(init)
assert not missing, f"missing head weights: {missing}"
np.savez_compressed(BUNDLE / "explain_head.npz",
                    **{k.replace("model.", "").replace(".", "_"): v.astype(np.float32) for k, v in init.items()})

onnx.checker.check_model(model)
onnx.save(model, str(BUNDLE / "model.onnx"))
print("model.onnx outputs:", [o.name for o in graph.output])
print("explain_head.npz:", sorted(init))
