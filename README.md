
# CKD Multimodal Prediction System

A complete AI-powered **Chronic Kidney Disease (CKD) prediction system** that fuses 
**Retinal Fundus Imaging (ResNet-18)** + **Clinical Biomarkers** using a multimodal deep-learning architecture.

This project includes:

- 🧠 **Multimodal ResNet18 + Clinical MLP Fusion Model** 
- 🩺 **Clinical Risk Analysis + Explainability** 
- 👁️ **Retinal Feature Extraction + Grad-CAM Heatmaps** 
- ⚙️ **Flask Backend API (TorchScript Inference)** 
- 📦 **Dataset Builder (Retina + CKD Dataset Fusion)** 
- 🎨 **Full Web Frontend (Upload + Predict + Heatmap)** 
- 🛠️ **Training Scripts + Checkpoint + TorchScript Export**

---

# 📁 Project Structure

```
CKD-Multimodal-Project/
│
├── backend/
│   ├── app.py                      # Flask API for prediction
│   ├── gradcam.py                  # Grad-CAM explainability
│   ├── models/
│   │   ├── checkpoint.pt           # Trained model weights
│   │   └── model.pt                # TorchScript model for backend
│   ├── uploads/                    # Uploaded images
│   └── xai_output/                 # Grad-CAM heatmaps
│
├── frontend/
│   ├── index.html
│   ├── predict.html
│   └── about.html
│
├── training/
│   ├── dataset.py                  # Multimodal dataset loader
│   ├── train_multimodal.py         # Full training pipeline
│   ├── export_model.py             # TorchScript model exporter
│
├── scripts/
│   └── build_multimodal.py         # Retina + CKD dataset merger
│
├── data_raw/
│   ├── retinal/                    # Raw retinal images (Kaggle)
│   └── ckd/                        # CKD clinical dataset
│
└── dataset/
    └── multimodal_dataset.csv      # Final multimodal dataset

```

---

# 🚀 Features

## 🔬 1. Multimodal Fusion
Image branch: **ResNet-18** 
Clinical branch: **MLP (11 features)** 
Fusion: **512 + 32 → CKD risk**

## 👁️ 2. Grad-CAM Explainability
- Heatmap over retinal fundus image 
- Highlights medically relevant regions 

## ⚙️ 3. Flask Backend API
Endpoints:
- `/predict`
- `/predict_image_only`

## 🎨 4. Frontend UI
- Upload retinal image
- Enter clinical biomarkers
- Get CKD risk + heatmap

---

# 🏋️ Training

```
python scripts/build_multimodal.py
python training/train_multimodal.py
python training/export_model.py
```

---

# ▶️ Running Backend

```
python backend/app.py
```

Access:
```
http://localhost:5000
```

---

# Installation

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt #outside_backend
```

---

# Requirements

```
flask
flask-cors
pandas
numpy
torch
torchvision
tqdm
pillow
opencv-python
scikit-learn
```

---
