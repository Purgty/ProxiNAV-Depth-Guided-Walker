import os
import cv2
import torch
import numpy as np

# ================= CONFIG =================

IMAGE_DIR = "Output2/plain_inference/images"
DEPTH_DIR = "Output2/plain_inference/depth_maps"

os.makedirs(DEPTH_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD MIDAS =================

print("[INFO] Loading MiDaS DPT_Hybrid...")

model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
model.to(DEVICE)
model.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform

print("[INFO] MiDaS model loaded successfully.")

# ================= PROCESS IMAGES =================

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

print(f"[INFO] Found {len(image_files)} images.")

with torch.no_grad():
    for idx, fname in enumerate(image_files):
        img_path = os.path.join(IMAGE_DIR, fname)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, _ = img.shape

        # Convert BGR → RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transform
        input_batch = transform(img_rgb).to(DEVICE)

        # Inference
        depth = model(input_batch)

        # Resize depth to original resolution
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = depth.cpu().numpy()

        # Normalize depth (near = high)
        depth_min = depth.min()
        depth_max = depth.max()

        if depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min)

        depth = np.clip(depth, 0.0, 1.0)

        # Save depth map
        out_path = os.path.join(DEPTH_DIR, fname.replace(".png", ".npy"))
        np.save(out_path, depth.astype(np.float32))

        if idx % 50 == 0:
            print(f"[INFO] Processed {idx}/{len(image_files)} images")

print("[DONE] Depth map generation completed.")