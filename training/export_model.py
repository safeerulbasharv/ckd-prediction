# training/export_model.py
import torch
from torch import nn
from torchvision import models
from pathlib import Path
import sys

CKPT_PATH = "backend/models/checkpoint.pt"
OUT_PATH = "backend/models/model.pt"

if not Path(CKPT_PATH).exists():
    print(f"ERROR: checkpoint not found at {CKPT_PATH}")
    sys.exit(1)

ckpt = torch.load(CKPT_PATH, map_location="cpu")
print("Loaded checkpoint keys:", list(ckpt.keys()))

# Try to infer clinical column count
clin_cols = ckpt.get("clin_cols", None)
if clin_cols is not None:
    num_clin = len(clin_cols)
    print(f"Using clin_cols from checkpoint: {num_clin} features")
else:
    # fallback: accept explicit numbers stored in checkpoint or default to 11
    # (11 is the default used in training scripts)
    num_clin = ckpt.get("num_clin", None) or 11
    print(f"No clin_cols in checkpoint; using num_clin={num_clin}")

# Utility to get state dict with multiple possible key names
def get_state(ckpt, *names):
    for n in names:
        if n in ckpt:
            return ckpt[n]
    return None

img_state = get_state(ckpt, "img_state", "img", "image_state", "model_img")
clin_state = get_state(ckpt, "clin_state", "clin", "clinical_state")
fusion_state = get_state(ckpt, "fusion_state", "fusion", "head_state", "fc_state")

if img_state is None:
    print("ERROR: Could not find image encoder state in checkpoint. Tried keys:",
          ["img_state","img","image_state","model_img"])
    sys.exit(1)
if clin_state is None:
    print("ERROR: Could not find clinical MLP state in checkpoint. Tried keys:",
          ["clin_state","clin","clinical_state"])
    sys.exit(1)
if fusion_state is None:
    print("ERROR: Could not find fusion head state in checkpoint. Tried keys:",
          ["fusion_state","fusion","head_state","fc_state"])
    sys.exit(1)

print("✔ Found model state dicts in checkpoint. Building model...")

# Build model matching training architecture
class MultiModalModel(nn.Module):
    def __init__(self, num_clin):
        super().__init__()
        # resnet backbone (same arch as training)
        try:
            # create resnet without pretrained weights
            res = models.resnet18(weights=None)
        except Exception:
            res = models.resnet18(pretrained=False)
        numf = res.fc.in_features
        res.fc = nn.Identity()
        self.img = res

        self.clin = nn.Sequential(
            nn.Linear(num_clin, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.fuse = nn.Sequential(
            nn.Linear(numf + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

    def forward(self, img, clin):
        a = self.img(img)
        b = self.clin(clin)
        x = torch.cat([a, b], dim=1)
        return self.fuse(x)

model = MultiModalModel(num_clin)

# load state dicts carefully
try:
    model.img.load_state_dict(img_state)
    model.clin.load_state_dict(clin_state)
    model.fuse.load_state_dict(fusion_state)
except Exception as e:
    print("State dict load failed:", e)
    print("Attempting to load partial keys / inspect shapes...")
    # show some helpful info
    print("img_state keys (sample):", list(img_state.keys())[:5])
    sys.exit(1)

model.eval()
print("✔ Model weights loaded into model instance.")

# Trace with example inputs
example_img = torch.randn(1, 3, 224, 224)
example_clin = torch.randn(1, num_clin)

print("Tracing TorchScript (this may take a few seconds)...")
try:
    scripted = torch.jit.trace(model, (example_img, example_clin))
    Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    scripted.save(OUT_PATH)
    print(f"✔ Exported TorchScript model to {OUT_PATH}")
except Exception as e:
    print("ERROR while tracing/saving TorchScript:", e)
    sys.exit(1)

