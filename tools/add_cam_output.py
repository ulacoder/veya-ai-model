"""Adds the last feature map of the un-flipped TTA branch as an extra ONNX output ("feat")
and saves the neck/head weights to cam_head.npz, so the API can compute Grad-CAM in numpy
(no PyTorch at runtime). Run once after exporting model.onnx:

    python tools/add_cam_output.py model_bundle/model.onnx
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

FEAT = "/model/backbone/bn2/act/Mul_output_0"  # un-flipped branch (second TTA pass has the "_1" suffix)
WEIGHTS = ["model.neck.1.weight", "model.neck.1.bias", "model.neck.4.weight", "model.neck.4.bias"] + [
    f"model.heads.{k}.{p}" for k in ("dr_grade", "dr", "glaucoma", "cataract") for p in ("weight", "bias")]


def main(path):
    path = Path(path)
    m = onnx.load(str(path))
    g = m.graph
    if any(o.name == "feat" for o in g.output):
        print("feat output already present")
    else:
        assert any(FEAT in n.output for n in g.node), "feature map tensor not found"
        g.node.append(helper.make_node("Identity", [FEAT], ["feat"], name="cam_feat"))
        g.output.append(helper.make_tensor_value_info("feat", TensorProto.FLOAT, None))
        onnx.save(m, str(path))
        print("added output 'feat'")
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    out = {k.replace("model.", "").replace(".", "_"): init[k].astype(np.float32) for k in WEIGHTS}
    np.savez(path.parent / "cam_head.npz", **out)
    print("saved", path.parent / "cam_head.npz", {k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "model_bundle/model.onnx")
