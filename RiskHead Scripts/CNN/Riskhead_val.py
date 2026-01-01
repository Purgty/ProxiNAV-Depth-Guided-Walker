import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import torch.nn as nn

# ================= CONFIG =================

BASE_DIR = "Output2/plain_inference"

OBS_DIR = f"{BASE_DIR}/obstacle_maps"
WALK_DIR = f"{BASE_DIR}/walkability_maps"
DEPTH_DIR = f"{BASE_DIR}/depth_maps"
IMG_DIR = f"{BASE_DIR}/images"

MODEL_PATH = "Riskhead Scripts/risk_cnn.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEST_FRAME = "image_000000.npy"  # choose one NOT used in training

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

X = np.stack([O, S, D], axis=0)
X = torch.from_numpy(X).unsqueeze(0).float().to(DEVICE)

# ================= INFERENCE =================

model = RiskCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with torch.no_grad():
    R_pred = model(X)[0, 0].cpu().numpy()

# ================= VISUALIZATION =================

img = cv2.imread(os.path.join(IMG_DIR, TEST_FRAME.replace(".npy", ".png")))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.title("RGB"); plt.imshow(img); plt.axis("off")
plt.subplot(1,3,2); plt.title("Predicted Risk"); plt.imshow(R_pred, cmap="inferno"); plt.axis("off")
plt.subplot(1,3,3); plt.title("Walkability"); plt.imshow(S, cmap="gray"); plt.axis("off")
plt.show()