import json, torch, timm
import torch.nn as nn
from PIL import Image
from torchvision import transforms

class VeyaFundusModel(nn.Module):
    def __init__(self, config):
        super().__init__(); self.targets=config["targets"]
        self.backbone=timm.create_model(config["backbone"], pretrained=False, num_classes=0)
        dim=self.backbone.num_features; self.heads=nn.ModuleDict()
        for name, kind in self.targets.items():
            hidden=512 if name=="dr_grade" else 256; out_dim=5 if kind=="multiclass" else 1
            self.heads[name]=nn.Sequential(nn.Dropout(0.25),nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(0.15),nn.Linear(hidden,out_dim))
    def forward(self,x):
        f=self.backbone(x); return {n:h(f) for n,h in self.heads.items()}

def load_model(bundle_dir=".", device=None):
    device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    with open(f"{bundle_dir}/model_config.json") as f: config=json.load(f)
    model=VeyaFundusModel(config)
    state=torch.load(f"{bundle_dir}/best_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state["model_state_dict"]); model.to(device).eval()
    del state
    return model, config, device

def predict(image_path, bundle_dir="."):
    model, config, device=load_model(bundle_dir)
    tfm=transforms.Compose([transforms.Resize((config["image_size"],config["image_size"])),transforms.ToTensor(),transforms.Normalize(config["mean"],config["std"])])
    image=Image.open(image_path).convert("RGB")
    with torch.no_grad(): outputs=model(tfm(image).unsqueeze(0).to(device))
    result={}
    for name, kind in config["targets"].items():
        if kind=="multiclass":
            probs=torch.softmax(outputs[name],1)[0].cpu().tolist(); result[name]={"class":int(torch.tensor(probs).argmax()),"probabilities":probs,"confidence":max(probs)}
        else:
            prob=float(torch.sigmoid(outputs[name].squeeze()).cpu()); result[name]={"probability":prob,"positive":prob>=0.5}
    return result
