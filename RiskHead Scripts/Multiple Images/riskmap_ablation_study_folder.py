import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path

# ================= CONFIG =================

BASE_DIR = "Output2/plain_inference"
MODEL_PATH = "Riskhead Scripts/CNN/risk_cnn.pt"

OBS_DIR = f"{BASE_DIR}/obstacle_maps"
WALK_DIR = f"{BASE_DIR}/walkability_maps"
DEPTH_DIR = f"{BASE_DIR}/depth_maps"
IMG_DIR = f"{BASE_DIR}/images"

OUTPUT_DIR = f"{BASE_DIR}/ablation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL =================

class RiskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ================= LOAD MODEL =================

print("[INFO] Loading RiskCNN...")
model = RiskCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("[INFO] Model loaded successfully.")

# ================= INFERENCE FUNCTION =================

def run_inference(O, S, D):
    X = np.stack([O, S, D], axis=0)
    X = torch.from_numpy(X).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        return model(X)[0, 0].cpu().numpy()

# ================= VISUALIZATION HELPER =================

def show_risk(ax, risk, title):
    im = ax.imshow(
        risk,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0
    )
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im

# ================= GET ALL FRAMES =================

npy_files = sorted(Path(OBS_DIR).glob("*.npy"))
if not npy_files:
    raise RuntimeError(f"No .npy files found in {OBS_DIR}")

print(f"[INFO] Found {len(npy_files)} frames")

# ================= PROCESS FRAMES =================

for idx, npy_path in enumerate(npy_files):
    fname = npy_path.name
    print(f"\n[INFO] Processing {idx+1}/{len(npy_files)}: {fname}")

    try:
        O = np.load(os.path.join(OBS_DIR, fname))
        S = np.load(os.path.join(WALK_DIR, fname))
        D = np.load(os.path.join(DEPTH_DIR, fname))

        img_name = fname.replace(".npy", ".png")
        img = cv2.imread(os.path.join(IMG_DIR, img_name))
        if img is None:
            print("[WARNING] Image missing, skipping")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    except Exception as e:
        print(f"[WARNING] Data load failed: {e}")
        continue

    # ---------- Ablations ----------
    R_full = run_inference(O, S, D)
    R_no_obs = run_inference(np.zeros_like(O), S, D)
    R_no_walk = run_inference(O, np.zeros_like(S), D)
    R_no_depth = run_inference(O, S, np.zeros_like(D))

    # ---------- Plot ----------
    fig = plt.figure(figsize=(14, 8))

    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(img)
    ax1.set_title("RGB Frame")
    ax1.axis("off")

    ax2 = plt.subplot(2, 3, 2)
    im = show_risk(ax2, R_full, "Full RiskCNN")

    ax3 = plt.subplot(2, 3, 3)
    show_risk(ax3, R_no_obs, "No Obstacle Prior")

    ax4 = plt.subplot(2, 3, 5)
    show_risk(ax4, R_no_walk, "No Walkability Prior")

    ax5 = plt.subplot(2, 3, 6)
    show_risk(ax5, R_no_depth, "No Depth Prior")

    # ---------- Shared Colorbar ----------
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Navigation Risk Intensity", rotation=270, labelpad=15)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels([
        "Safe",
        "Low Risk",
        "Moderate",
        "High Risk",
        "Critical"
    ])

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    out_name = fname.replace(".npy", "_ablation.png")
    plt.savefig(os.path.join(OUTPUT_DIR, out_name), dpi=150)
    plt.close()

    print(f"[DONE] Saved: {out_name}")

print(f"\n[COMPLETE] All results saved to: {OUTPUT_DIR}")