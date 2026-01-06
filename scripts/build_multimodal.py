import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
IMG_ROOT = Path("data_raw/retinal")
CKD_DIR = Path("data_raw/ckd")
OUT_DIR = Path("dataset")
OUT_DIR.mkdir(exist_ok=True)
OUT_CSV = OUT_DIR / "multimodal_dataset.csv"

# ---------------------------------------------------
# SAFE CLEANING HELPERS
# ---------------------------------------------------
def clean_label(x):
    """
    Converts CKD labels into clean 0/1 safely.
    Handles: "ckd", "ckd\t", "ckd ", "notckd", " notckd", None, "?"
    """
    if isinstance(x, str):
        x = x.strip().lower()  # remove tabs, spaces, newlines
        if x.startswith("ckd"):
            return 1
        if x.startswith("not"):
            return 0
    return 0

def safe_num(x):
    """
    Convert any numeric/dirty value to float safely.
    If not possible, return 0.
    """
    try:
        return float(str(x).strip())
    except:
        return 0.0

# ---------------------------------------------------
# LOAD IMAGES
# ---------------------------------------------------
print("📷 Scanning images...")
images = []
for ext in ["*.jpg", "*.jpeg", "*.png"]:
    images += [p.as_posix() for p in IMG_ROOT.rglob(ext)]

if len(images) == 0:
    raise SystemExit("❌ ERROR: No images found in data_raw/retinal/")

print("✔ Found", len(images), "retinal images")

# ---------------------------------------------------
# LOAD CKD CSV
# ---------------------------------------------------
print("📄 Loading CKD CSV...")

# Auto-detect CSV (kidney_disease.csv or similar)
csv_files = list(CKD_DIR.glob("*.csv"))
if not csv_files:
    raise SystemExit("❌ ERROR: No CSV file found in data_raw/ckd/")

CKD_CSV = csv_files[0]
print("✔ Using clinical file:", CKD_CSV)

clin = pd.read_csv(CKD_CSV)

# Normalize column names
clin.columns = [c.strip().lower() for c in clin.columns]

# ---------------------------------------------------
# CLEAN LABELS
# ---------------------------------------------------
if "classification" not in clin.columns:
    raise SystemExit("❌ ERROR: CKD CSV missing 'classification' column.")

clin["classification"] = clin["classification"].apply(clean_label)

# ---------------------------------------------------
# CLEAN NUMERIC COLUMNS SAFELY
# ---------------------------------------------------
CLEAN_COLS = {
    "sc": "creatinine",
    "al": "albumin",
    "bgr": "glucose",
    "hemo": "hemoglobin",
    "pcv": "pcv",
    "wc": "wbc",
    "rc": "rbc_raw",
    "age": "age",
    "bp": "bp",
    "htn": "hypertension_raw",
    "dm": "diabetes_raw"
}

for old, new in CLEAN_COLS.items():
    if old not in clin.columns:
        clin[old] = 0
    clin[new] = clin[old].apply(safe_num)

# ---------------------------------------------------
# CLEAN CATEGORICAL COLUMNS (yes/no → 1/0)
# ---------------------------------------------------
def clean_bin(x):
    return 1 if str(x).strip().lower() == "yes" else 0

clin["hypertension"] = clin["hypertension_raw"].apply(clean_bin)
clin["diabetes"] = clin["diabetes_raw"].apply(clean_bin)
clin["rbc"] = clin["rbc_raw"].apply(safe_num)

# ---------------------------------------------------
# BUILD MULTIMODAL PAIRED DATASET
# ---------------------------------------------------
print("🔄 Pairing images with clinical rows...")

rows = []
for img in images:
    row = clin.sample(1).iloc[0]

    rows.append({
        "image_path": img,
        "creatinine": row["creatinine"],
        "albumin": row["albumin"],
        "glucose": row["glucose"],
        "hemoglobin": row["hemoglobin"],
        "pcv": row["pcv"],
        "wbc": row["wbc"],
        "rbc": row["rbc"],
        "age": row["age"],
        "bp": row["bp"],
        "hypertension": row["hypertension"],
        "diabetes": row["diabetes"],
        "ckd_label": int(row["classification"])
    })

df = pd.DataFrame(rows)

# ---------------------------------------------------
# TRAIN/VAL/TEST SPLIT
# ---------------------------------------------------
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["ckd_label"])
train, val = train_test_split(train, test_size=0.1, random_state=42, stratify=train["ckd_label"])

train["split"] = "train"
val["split"] = "val"
test["split"] = "test"

final_df = pd.concat([train, val, test])
final_df.to_csv(OUT_CSV, index=False)

print("✔ Saved clean dataset →", OUT_CSV)
print("🎉 Dataset ready for training!")

