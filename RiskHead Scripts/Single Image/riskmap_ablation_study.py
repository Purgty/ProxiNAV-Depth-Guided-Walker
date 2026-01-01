import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn

# ================= CONFIG =================

BASE_DIR = "Output/plain_inference"
MODEL_PATH = "Riskhead Scripts/risk_cnn.pt"

OBS_DIR = f"{BASE_DIR}/obstacle_maps"
WALK_DIR = f"{BASE_DIR}/walkability_maps"
DEPTH_DIR = f"{BASE_DIR}/depth_maps"
IMG_DIR = f"{BASE_DIR}/images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_FRAME = "frame_000000.npy"  # choose a representative frame

# ================= MODEL =================

class RiskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # MEMORY OPTIMIZATION: Slightly reduced channels
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),      # 32 -> 24
            nn.ReLU(inplace=True),                # inplace saves memory
            nn.Conv2d(24, 48, 3, padding=1),     # 64 -> 48
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),     # 64 -> 48
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 24, 3, padding=1),     # 32 -> 24
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ================= LOAD DATA =================

O = np.load(os.path.join(OBS_DIR, TEST_FRAME))
S = np.load(os.path.join(WALK_DIR, TEST_FRAME))
D = np.load(os.path.join(DEPTH_DIR, TEST_FRAME))

img = cv2.imread(os.path.join(IMG_DIR, TEST_FRAME.replace(".npy", ".png")))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def run_inference(O, S, D):
    X = np.stack([O, S, D], axis=0)
    X = torch.from_numpy(X).unsqueeze(0).float().to(DEVICE)
    with torch.no_grad():
        return model(X)[0,0].cpu().numpy()

# ================= LOAD MODEL =================

model = RiskCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ================= RUN ABLATIONS =================

R_full = run_inference(O, S, D)
R_no_obs = run_inference(np.zeros_like(O), S, D)
R_no_walk = run_inference(O, np.zeros_like(S), D)
R_no_depth = run_inference(O, S, np.zeros_like(D))

# ================= VISUALIZE =================

plt.figure(figsize=(14,8))

plt.subplot(2,3,1); plt.title("RGB"); plt.imshow(img); plt.axis("off")
plt.subplot(2,3,2); plt.title("Full Risk"); plt.imshow(R_full, cmap="inferno"); plt.axis("off")
plt.subplot(2,3,3); plt.title("No Obstacles"); plt.imshow(R_no_obs, cmap="inferno"); plt.axis("off")

plt.subplot(2,3,5); plt.title("No Walkability"); plt.imshow(R_no_walk, cmap="inferno"); plt.axis("off")
plt.subplot(2,3,6); plt.title("No Depth"); plt.imshow(R_no_depth, cmap="inferno"); plt.axis("off")

plt.tight_layout()
plt.show()