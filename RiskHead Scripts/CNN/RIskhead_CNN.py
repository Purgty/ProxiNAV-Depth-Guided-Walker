import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import matplotlib.pyplot as plt
import random

# ================= CONFIG =================

BASE_DIR = "Output/plain_inference"

OBS_DIR = f"{BASE_DIR}/obstacle_maps"
WALK_DIR = f"{BASE_DIR}/walkability_maps"
DEPTH_DIR = f"{BASE_DIR}/depth_maps"
RISK_DIR = f"{BASE_DIR}/risk_labels"
IMG_DIR = f"{BASE_DIR}/images"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
EPOCHS = 25
LR = 1e-4

# ================= DATASET =================

class RiskDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        O = np.load(os.path.join(OBS_DIR, fname))
        S = np.load(os.path.join(WALK_DIR, fname))
        D = np.load(os.path.join(DEPTH_DIR, fname))
        R = np.load(os.path.join(RISK_DIR, fname))

        X = np.stack([O, S, D], axis=0)
        Y = R[None, ...]

        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(Y).float(),
            fname
        )

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

# ================= TRAINING =================

files = sorted(os.listdir(RISK_DIR))
random.shuffle(files)

n = len(files)
train_files = files[:int(0.7 * n)]
val_files   = files[int(0.7 * n):int(0.85 * n)]
test_files  = files[int(0.85 * n):]

train_loader = DataLoader(RiskDataset(train_files), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(RiskDataset(val_files), batch_size=1)

model = RiskCNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print("[INFO] Starting training...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0

    for X, Y, _ in train_loader:
        X = X.to(DEVICE)
        Y = Y.to(DEVICE)

        pred = model(X)
        loss = criterion(pred, Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "risk_cnn.pt")
print("[DONE] RiskCNN trained and saved.")

# ================= SANITY CHECK =================

print("[INFO] Running sanity check...")

model.eval()
sample_files = random.sample(test_files, min(5, len(test_files)))

with torch.no_grad():
    for fname in sample_files:
        X = np.stack([
            np.load(os.path.join(OBS_DIR, fname)),
            np.load(os.path.join(WALK_DIR, fname)),
            np.load(os.path.join(DEPTH_DIR, fname))
        ], axis=0)

        X = torch.from_numpy(X).unsqueeze(0).float().to(DEVICE)
        pred = model(X)[0,0].cpu().numpy()
        gt = np.load(os.path.join(RISK_DIR, fname))

        img = cv2.imread(os.path.join(IMG_DIR, fname.replace(".npy", ".png")))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1); plt.title("RGB"); plt.imshow(img); plt.axis("off")
        plt.subplot(1,3,2); plt.title("GT Risk"); plt.imshow(gt, cmap="inferno"); plt.axis("off")
        plt.subplot(1,3,3); plt.title("Pred Risk"); plt.imshow(pred, cmap="inferno"); plt.axis("off")
        plt.show()