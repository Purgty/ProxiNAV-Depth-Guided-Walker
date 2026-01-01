import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random

# ================= CONFIG =================

BASE_DIR = "Output2/plain_inference"

IMG_DIR = f"{BASE_DIR}/images"
OBS_DIR = f"{BASE_DIR}/obstacle_maps"
WALK_DIR = f"{BASE_DIR}/walkability_maps"
DEPTH_DIR = f"{BASE_DIR}/depth_maps"
RISK_DIR = f"{BASE_DIR}/risk_labels"

os.makedirs(RISK_DIR, exist_ok=True)

# Risk weights (start values)
ALPHA = 0.45   # obstacle weight
BETA  = 0.35   # depth weight
GAMMA = 0.20   # walkability penalty

SMOOTH_SIGMA = 1.5   # spatial smoothing

# ================= LOAD FRAME LIST =================

files = sorted([
    f for f in os.listdir(OBS_DIR)
    if f.endswith(".npy")
])

print(f"[INFO] Found {len(files)} frames for risk generation.")

# ================= RISK GENERATION =================

for idx, fname in enumerate(files):
    O = np.load(os.path.join(OBS_DIR, fname))
    S = np.load(os.path.join(WALK_DIR, fname))
    D = np.load(os.path.join(DEPTH_DIR, fname))

    # Safety check
    assert O.shape == S.shape == D.shape, "Shape mismatch!"

    # Risk formulation
    R_hat = (
        ALPHA * O +
        BETA  * D +
        GAMMA * (1.0 - S)
    )

    # Clip and smooth
    R_hat = np.clip(R_hat, 0.0, 1.0)
    R_hat = gaussian_filter(R_hat, sigma=SMOOTH_SIGMA)

    # Save
    np.save(os.path.join(RISK_DIR, fname), R_hat.astype(np.float32))

    if idx % 50 == 0:
        print(f"[INFO] Generated risk for {idx}/{len(files)} frames")

print("[DONE] Risk label generation complete.")

# ================= SANITY CHECK =================

print("[INFO] Running sanity check visualization...")

sample_files = random.sample(files, min(4, len(files)))

for fname in sample_files:
    img = cv2.imread(os.path.join(IMG_DIR, fname.replace(".npy", ".png")))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    R = np.load(os.path.join(RISK_DIR, fname))

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.title("RGB")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("Risk Map")
    plt.imshow(R, cmap="inferno")
    plt.axis("off")

    plt.show()
