import os
import time
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import cv2
from gradcam import GradCAM, overlay_heatmap
import subprocess
import json

app = Flask(__name__)
CORS(app)

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD = os.path.join(BASE, "uploads")
XAI = os.path.join(BASE, "xai_output")
MODEL_TS_PATH = os.path.join(BASE, "models", "model.pt")           # TorchScript (optional)
CKPT_PATH = os.path.join(BASE, "models", "checkpoint.pt")         # state_dict checkpoint (preferred)

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(XAI, exist_ok=True)

# -------- safe parsing helper ----------

def safe_parse_float(x, default=0.0):
    """
    Accepts None, '', whitespace, 'NaN', strings and returns a float.
    """
    try:
        if x is None:
            return float(default)
        s = str(x).strip()
        if s == "":
            return float(default)
        if s.lower() in ["nan", "none", "?"]:
            return float(default)
        return float(s)
    except Exception:
        return float(default)

def parse_binary_flag(x):
    if x is None:
        return 0.0
    s = str(x).strip().lower()
    if s in ["1","yes","y","true","t"]:
        return 1.0
    return 0.0
    
@app.route("/llm_generate", methods=["POST"])
def llm_generate():
    try:
        # Gather model output from main prediction
        risk = request.form.get("risk", "")
        label = request.form.get("label", "")
        clinical = request.form.get("clinical_summary", "")

        prompt = f"""
        You are a medical assistant. Summarize CKD risk based on:
        - Risk score: {risk}
        - Category: {label}
        - Clinical summary: {clinical}

        Provide a patient-friendly explanation.
        Keep it short (1-3 lines).
        """

        # Call Ollama Phi-3 locally
        result = subprocess.run(
            ["ollama", "run", "llama3.2:1b", prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore"
        )

        llm_output = result.stdout.strip()

        return {"llm_text": llm_output}

    except Exception as e:
        return {"llm_text": f"LLM generation failed: {str(e)}"}
# -------- Model builder (must match training/export) ----------
import torch.nn as nn
from torchvision import models

class MultiModalModel(nn.Module):
    def __init__(self, num_clin=11):
        super().__init__()
        # build resnet18 backbone (no pretrained weights here)
        try:
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

# -------- Load model (prefer checkpoint with state_dict) ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
loaded_from = None

if os.path.exists(CKPT_PATH):
    try:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        clin_cols = ckpt.get("clin_cols", None)
        num_clin = len(clin_cols) if clin_cols else ckpt.get("num_clin", 11)
        model = MultiModalModel(num_clin=num_clin)
        # load keys tolerant to names used earlier
        def get_state(d, *names):
            for n in names:
                if n in d:
                    return d[n]
            return None
        img_state = get_state(ckpt, "img_state", "img")
        clin_state = get_state(ckpt, "clin_state", "clin")
        fusion_state = get_state(ckpt, "fusion_state", "fusion")

        if img_state is None or clin_state is None or fusion_state is None:
            print("Checkpoint exists but missing pieces; falling back to TorchScript if available.")
            model = None
        else:
            model.img.load_state_dict(img_state)
            model.clin.load_state_dict(clin_state)
            model.fuse.load_state_dict(fusion_state)
            model.to(device)
            model.eval()
            loaded_from = "checkpoint"
            print("Loaded model from checkpoint.pt")
    except Exception as e:
        print("Failed to load checkpoint.pt:", e)
        model = None

if model is None and os.path.exists(MODEL_TS_PATH):
    try:
        model = torch.jit.load(MODEL_TS_PATH, map_location=device)
        model.to(device)
        model.eval()
        loaded_from = "torchscript"
        print("Loaded model from TorchScript model.pt (fallback)")
    except Exception as e:
        print("Failed to load TorchScript model:", e)
        model = None

if model is None:
    print("WARNING: No model loaded. The API will return dummy predictions.")
else:
    print(f"Model loaded from: {loaded_from}, device: {device}")

# -------- transforms ----------
preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def read_image_as_tensor_and_rgb(path):
    img = Image.open(path).convert("RGB")
    rgb = np.array(img)  # H,W,3 (RGB)
    tensor = preprocess(img).unsqueeze(0)
    return tensor, rgb

# -------- PREDICT (image + clinical) ----------
@app.route("/predict", methods=["POST"])
def predict():
    # ensure file exists
    if "retinal_image" not in request.files:
        return jsonify({"error":"retinal_image is required"}), 400

    img_file = request.files["retinal_image"]
    if img_file.filename == "":
        return jsonify({"error":"No filename provided"}), 400

    save_name = f"{int(time.time())}_{img_file.filename}"
    save_path = os.path.join(UPLOAD, save_name)
    img_file.save(save_path)

    # parse clinical features safely
    creatinine = safe_parse_float(request.form.get("creatinine", ""), 0.0)
    albumin = safe_parse_float(request.form.get("albumin", ""), 0.0)
    glucose = safe_parse_float(request.form.get("glucose", ""), 0.0)
    hemoglobin = safe_parse_float(request.form.get("hemoglobin", ""), 0.0)
    pcv = safe_parse_float(request.form.get("pcv", ""), 0.0)
    wbc = safe_parse_float(request.form.get("wbc", ""), 0.0)
    rbc = safe_parse_float(request.form.get("rbc", ""), 0.0)
    age = safe_parse_float(request.form.get("age", ""), 0.0)
    bp = safe_parse_float(request.form.get("bp", ""), 0.0)
    hypertension = parse_binary_flag(request.form.get("hypertension", "0"))
    diabetes = parse_binary_flag(request.form.get("diabetes", "0"))

    clin_list = [
        creatinine, albumin, glucose, hemoglobin,
        pcv, wbc, rbc, age, bp, hypertension, diabetes
    ]

    # build tensors
    img_tensor, orig_rgb = read_image_as_tensor_and_rgb(save_path)
    clin_tensor = torch.tensor(clin_list, dtype=torch.float32).unsqueeze(0)

    # run model or fallback
    if model is None:
        # dummy response
        heat_name = None
        resp = {
            "final_risk": 50,
            "risk_label": "Moderate Risk",
            "clinical_contribution": 40,
            "retinal_contribution": 60,
            "clinical_details": [f"Age: {age}", f"Creatinine: {creatinine}", f"Albumin: {albumin}"],
            "xai_image_url": None,
            "xai_summary": "No model loaded - placeholder."
        }
        return jsonify(resp)

    # make sure tensors on same device
    img_tensor = img_tensor.to(device)
    clin_tensor = clin_tensor.to(device)

    # forward
    with torch.no_grad():
        try:
            # model may be a scripted module that expects (img, clin) OR original nn.Module
            out = model(img_tensor, clin_tensor)
            if isinstance(out, torch.Tensor):
                score = torch.sigmoid(out).cpu().item()
            else:
                # if model returns list/tuple
                score = float(torch.sigmoid(out[0]).cpu().item())
        except Exception as e:
            # If direct forward fails (for scripted models that require separate call),
            # try calling only with img to get a scalar (fallback)
            print("Model forward failed:", e)
            return jsonify({"error": f"Model forward failed: {str(e)}"}), 500

    # grad-cam (only if model is a full nn.Module with attribute .img)
    heat_name = None
    try:
        if hasattr(model, "img"):  # full nn.Module - GradCAM works
            gc = GradCAM(model)
            cam = gc.generate(img_tensor)
            overlay = overlay_heatmap(orig_rgb, cam)
            heat_name = "heat_" + img_file.filename.replace(".", "_") + ".jpg"
            heat_path = os.path.join(XAI, heat_name)
            cv2.imwrite(heat_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            xai_url = request.host_url.rstrip("/") + "/xai_output/" + heat_name
        else:
            # scripted model: cannot reliably compute GradCAM (fallback)
            xai_url = None
    except Exception as e:
        print("GradCAM/heatmap failed:", e)
        xai_url = None

    risk_pct = int(round(score * 100))
    label = "Low Risk" if risk_pct < 30 else "Moderate Risk" if risk_pct < 70 else "High Risk"

    resp = {
        "final_risk": risk_pct,
        "risk_label": label,
        "clinical_contribution": 40,
        "retinal_contribution": 60,
        "clinical_details": [
            f"Age: {age}",
            f"Creatinine: {creatinine}",
            f"Albumin: {albumin}",
        ],
        "xai_image_url": xai_url,
        "xai_summary": "Grad-CAM generated." if xai_url else "Grad-CAM not available.",
        "rule_based": "Automated multimodal interpretation."
    }
    return jsonify(resp)

# -------- PREDICT IMAGE ONLY ----------
@app.route("/predict_image_only", methods=["POST"])
def predict_image_only():
    if "retinal_image" not in request.files:
        return jsonify({"error":"retinal_image is required"}), 400
    img_file = request.files["retinal_image"]
    if img_file.filename == "":
        return jsonify({"error":"No filename provided"}), 400

    save_name = f"{int(time.time())}_{img_file.filename}"
    save_path = os.path.join(UPLOAD, save_name)
    img_file.save(save_path)

    img_tensor, orig_rgb = read_image_as_tensor_and_rgb(save_path)
    img_tensor = img_tensor.to(device)

    if model is None:
        return jsonify({
            "final_risk": 50,
            "risk_label": "Moderate Risk",
            "xai_image_url": None,
            "xai_summary": "No model loaded."
        })

    # Try forward - some scripted models accept (img,clin) only; we call with dummy clin if needed
    try:
        # create dummy clinical tensor sized based on model.clin if available
        if hasattr(model, "clin"):
            num_clin = list(model.clin.children())[0].in_features if hasattr(model.clin, "children") else 11
            clin_dummy = torch.zeros(1, num_clin, dtype=torch.float32).to(device)
            out = model(img_tensor, clin_dummy)
        else:
            # scripted fallback: try calling with img only
            out = model(img_tensor, torch.zeros(1,11).to(device))
        score = torch.sigmoid(out).cpu().item()
    except Exception as e:
        print("Predict image-only forward failed:", e)
        return jsonify({"error": f"prediction failed: {str(e)}"}), 500

    # Grad-CAM (only for full nn.Module)
    xai_url = None
    try:
        if hasattr(model, "img"):
            gc = GradCAM(model)
            cam = gc.generate(img_tensor)
            overlay = overlay_heatmap(orig_rgb, cam)
            heat_name = "heat_" + img_file.filename.replace(".", "_") + ".jpg"
            heat_path = os.path.join(XAI, heat_name)
            cv2.imwrite(heat_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            xai_url = request.host_url.rstrip("/") + "/xai_output/" + heat_name
    except Exception as e:
        print("GradCAM failed for image-only:", e)

    risk_pct = int(round(score * 100))
    label = "Low Risk" if risk_pct < 30 else "Moderate Risk" if risk_pct < 70 else "High Risk"

    return jsonify({
        "final_risk": risk_pct,
        "risk_label": label,
        "xai_image_url": xai_url,
        "xai_summary": "Grad-CAM generated." if xai_url else "Grad-CAM not available."
    })


# -------- Serve heatmap ----------
@app.route("/xai_output/<path:name>")
def serve_xai(name):
    return send_from_directory(XAI, name)

@app.route("/")
def home():
    return jsonify({
        "status": "CKD Multimodal AI Backend Running 🚀",
        "endpoints": [
            "POST /predict",
            "POST /predict_image_only",
            "POST /llm_generate"
        ]
    })

if __name__ == "__main__":
    print("➡️ Loading model...")
    print(f"Model object loaded: {model is not None}")
    print("🚀 Backend running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

