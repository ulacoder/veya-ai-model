"""Export model_bundle/best_model.pth to model_bundle/model.onnx (single file).

Run once after retraining:  python tools/export_onnx.py
"""
import sys
from pathlib import Path

import torch

BUNDLE = Path(__file__).resolve().parent.parent / "model_bundle"
sys.path.insert(0, str(BUNDLE))
from inference import load_model  # noqa: E402


class Wrapper(torch.nn.Module):
    def __init__(self, model, names):
        super().__init__()
        self.model, self.names = model, names

    def forward(self, x):
        out = self.model(x)
        return tuple(out[n] for n in self.names)


model, config, _ = load_model(str(BUNDLE), "cpu")
names = list(config["targets"])
size = int(config["image_size"])
torch.onnx.export(
    Wrapper(model, names).eval(),
    torch.randn(1, 3, size, size),
    str(BUNDLE / "model.onnx"),
    input_names=["image"],
    output_names=names,
    opset_version=17,
    dynamo=False,
)
print("saved", BUNDLE / "model.onnx", "outputs:", names)
