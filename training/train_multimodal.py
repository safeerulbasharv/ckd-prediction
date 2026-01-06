# training/train_multimodal.py
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet18_Weights
from dataset import MultimodalDataset, CLIN_COLS
from tqdm import tqdm
import numpy as np

# -------- CONFIG --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = torch.cuda.is_available()  # enable AMP only if CUDA available
CSV = "dataset/multimodal_dataset.csv"
BATCH = 16
EPOCHS = 8
LR = 1e-5           # lower LR to improve stability
OUT_PATH = "backend/models/checkpoint.pt"
GRAD_CLIP = 2.0     # gradient clipping norm
LOGIT_CLAMP = 50.0  # clamp logits to avoid overflow

print(f"▶ Device: {DEVICE}")
print(f"▶ AMP Enabled: {USE_AMP}")
print(f"▶ Clinical feature count: {len(CLIN_COLS)}")

# -------- DATA --------
train_ds = MultimodalDataset(CSV, "train")
val_ds = MultimodalDataset(CSV, "val")

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

print(f"✔ Train samples: {len(train_ds)}")
print(f"✔ Val samples:   {len(val_ds)}")

# Quick sample check (debug)
try:
    s_img, s_clin, s_lbl = train_ds[0]
    print("Sample clinical (first):", s_clin.numpy())
    print("Sample label (first):", s_lbl.item())
except Exception as e:
    print("Warning: could not show sample (dataset issue):", e)

# -------- MODEL --------
res = models.resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = res.fc.in_features
res.fc = nn.Identity()

clin_mlp = nn.Sequential(
    nn.Linear(len(CLIN_COLS), 64),
    nn.ReLU(),
    nn.BatchNorm1d(64),
    nn.Linear(64, 32),
    nn.ReLU()
)

fusion = nn.Sequential(
    nn.Linear(num_features + 32, 128),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(128, 1)
)

res = res.to(DEVICE)
clin_mlp = clin_mlp.to(DEVICE)
fusion = fusion.to(DEVICE)

# -------- OPTIMIZER / LOSS / AMP / CLIP --------
optimizer = torch.optim.Adam(
    list(res.parameters()) + list(clin_mlp.parameters()) + list(fusion.parameters()),
    lr=LR
)
criterion = nn.BCEWithLogitsLoss()

if USE_AMP:
    from torch.cuda.amp import GradScaler, autocast
    scaler = GradScaler()
else:
    # dummy autocast context manager for CPU
    from contextlib import nullcontext
    autocast = nullcontext
    class DummyScaler:
        def scale(self, loss): return loss
        def step(self, opt): opt.step()
        def update(self): pass
    scaler = DummyScaler()

# -------- TRAIN LOOP --------
for epoch in range(1, EPOCHS + 1):
    print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
    res.train(); clin_mlp.train(); fusion.train()

    total_loss = 0.0
    step = 0
    train_bar = tqdm(train_dl, desc="Training", ncols=100)

    for img, clin, label in train_bar:
        img = img.to(DEVICE)
        clin = clin.to(DEVICE)
        label = label.to(DEVICE).unsqueeze(1)

        optimizer.zero_grad()

        with (autocast() if USE_AMP else torch.no_grad() if False else torch.no_grad()):
            # Note: on CPU autocast is nullcontext, safe to call
            # but to keep API consistent, use the following:
            if USE_AMP:
                ctx = autocast()
            else:
                ctx = autocast()

        # Use a regular with-block for both cases
        with ctx:
            img_feat = res(img)
            clin_feat = clin_mlp(clin)
            fused = torch.cat([img_feat, clin_feat], dim=1)
            output = fusion(fused)
            # clamp logits to avoid huge values leading to NaN
            output = torch.clamp(output, -LOGIT_CLAMP, LOGIT_CLAMP)
            loss = criterion(output, label)

        # backward + step (with or without scaler)
        if USE_AMP:
            scaler.scale(loss).backward()
            # gradient clipping with scaler
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(res.parameters()) + list(clin_mlp.parameters()) + list(fusion.parameters()), GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(res.parameters()) + list(clin_mlp.parameters()) + list(fusion.parameters()), GRAD_CLIP
            )
            optimizer.step()

        total_loss += loss.item()
        step += 1
        train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_train_loss = total_loss / max(1, step)

    # -------- VALIDATION --------
    res.eval(); clin_mlp.eval(); fusion.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_bar = tqdm(val_dl, desc="Validating", ncols=100)
    with torch.no_grad():
        for img, clin, label in val_bar:
            img = img.to(DEVICE)
            clin = clin.to(DEVICE)
            label = label.to(DEVICE).unsqueeze(1)

            if USE_AMP:
                with autocast():
                    img_feat = res(img)
                    clin_feat = clin_mlp(clin)
                    fused = torch.cat([img_feat, clin_feat], dim=1)
                    output = fusion(fused)
            else:
                img_feat = res(img)
                clin_feat = clin_mlp(clin)
                fused = torch.cat([img_feat, clin_feat], dim=1)
                output = fusion(fused)

            output = torch.clamp(output, -LOGIT_CLAMP, LOGIT_CLAMP)
            loss = criterion(output, label)
            val_loss += loss.item()

            preds = (torch.sigmoid(output) > 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)
            val_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_val_loss = val_loss / max(1, len(val_dl))
    val_acc = correct / max(1, total)

    print(f"Epoch {epoch} summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

# -------- SAVE CHECKPOINT --------
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
torch.save({
    "img_state": res.state_dict(),
    "clin_state": clin_mlp.state_dict(),
    "fusion_state": fusion.state_dict(),
    "clin_cols": CLIN_COLS
}, OUT_PATH)

print(f"\n✔ Saved checkpoint to {OUT_PATH}")

