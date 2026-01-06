# training/dataset.py
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Clinical columns used by the model (order matters)
CLIN_COLS = [
    'creatinine', 'albumin', 'glucose', 'hemoglobin',
    'pcv', 'wbc', 'rbc', 'age', 'bp', 'hypertension', 'diabetes'
]

# Image transform
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# SAFE numeric parser
def safe_num(x, default=0.0):
    try:
        # handle strings with commas/tabs and missing values
        if pd.isna(x):
            return float(default)
        s = str(x).strip().replace(',', '').replace('\t', '')
        if s in ['', '?', 'na', 'none', 'nan']:
            return float(default)
        return float(s)
    except Exception:
        return float(default)

# sensible per-feature min/max for min-max scaling (used to normalize clinical features)
# These ranges are conservative, tune them if you know your data range.
# Order must match CLIN_COLS.
FEATURE_MIN = torch.tensor([0.0, 0.0, 0.0, 5.0, 10.0, 2000.0, 2.0, 0.0, 40.0, 0.0, 0.0], dtype=torch.float32)
FEATURE_MAX = torch.tensor([15.0, 5.0, 500.0, 18.0, 60.0, 18000.0, 6.0, 100.0, 200.0, 1.0, 1.0], dtype=torch.float32)
EPS = 1e-8

class MultimodalDataset(Dataset):
    def __init__(self, csv_path, split="train"):
        df = pd.read_csv(csv_path)
        if 'split' in df.columns:
            self.df = df[df["split"] == split].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        # Ensure all clinical columns exist; fill missing with 0
        for c in CLIN_COLS:
            if c not in self.df.columns:
                self.df[c] = 0.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = row["image_path"]
        img = Image.open(img_path).convert("RGB")
        img = transform(img)

        # Build clinical tensor (safe parsing)
        values = []
        for c in CLIN_COLS:
            v = safe_num(row.get(c, 0.0), default=0.0)
            values.append(v)
        clin = torch.tensor(values, dtype=torch.float32)

        # Clip extreme outliers and min-max normalize into [0,1]
        clin = torch.clamp(clin, FEATURE_MIN, FEATURE_MAX)
        clin = (clin - FEATURE_MIN) / (FEATURE_MAX - FEATURE_MIN + EPS)
        clin = torch.clamp(clin, 0.0, 1.0)

        # Label (safe)
        raw_label = row.get("ckd_label", 0)
        # handle any stray strings like 'ckd\t' or ' notckd'
        if isinstance(raw_label, str):
            label = 1.0 if str(raw_label).strip().lower().startswith("ckd") else 0.0
        else:
            try:
                label = float(raw_label)
            except Exception:
                label = 0.0
        label = torch.tensor(label, dtype=torch.float32)

        return img, clin, label

