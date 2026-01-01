import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.ndimage import gaussian_filter

# ================= CONFIG =================

IMAGE_PATH = r"Assets\Pavement2.png"  # <-- single image

YOLO_DET_WEIGHTS = r"WOTR_Dataset_Model\best.pt"
YOLO_SEG_WEIGHTS = r"Pavement_Dataset_Model\train2\weights\best.pt"

CONF_THRESH = 0.25
BLUR_SIGMA = 3

# 🔴 IMPORTANT: adjust to your segmentation dataset
WALKABLE_CLASS_IDS = [2]  # e.g. sidewalk

BASE_DIR = "Output2/plain_inference"
IMG_DIR = f"{BASE_DIR}/images"
OBS_DIR = f"{BASE_DIR}/obstacle_maps"
SEG_DIR = f"{BASE_DIR}/walkability_maps"

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(OBS_DIR, exist_ok=True)
os.makedirs(SEG_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD MODELS =================

print("[INFO] Loading Ultralytics YOLO models...")

det_model = YOLO(YOLO_DET_WEIGHTS)
seg_model = YOLO(YOLO_SEG_WEIGHTS)

print("[INFO] Models loaded successfully.")

# ================= LOAD IMAGE =================

frame = cv2.imread(IMAGE_PATH)
assert frame is not None, f"Failed to load image: {IMAGE_PATH}"

h, w, _ = frame.shape
fname = "image_000000.png"

cv2.imwrite(os.path.join(IMG_DIR, fname), frame)

PAVEMENT_CLASS_ID = 0
PAVEMENT_CONF_THRESH = 0.40

# ================= OBSTACLE MAP =================

print("[INFO] Running obstacle detection...")

det_results = det_model(frame, conf=CONF_THRESH, verbose=False)[0]
O = np.zeros((h, w), dtype=np.float32)

if det_results.boxes is not None:
    boxes = det_results.boxes.xyxy.cpu().numpy()
    scores = det_results.boxes.conf.cpu().numpy()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        O[y1:y2, x1:x2] = np.maximum(O[y1:y2, x1:x2], score)

O = gaussian_filter(O, sigma=BLUR_SIGMA)
O = np.clip(O, 0.0, 1.0)

np.save(os.path.join(OBS_DIR, fname.replace(".png", ".npy")), O)

# ================= WALKABILITY MAP =================

print("[INFO] Running pavement segmentation...")

seg_results = seg_model(frame, conf=PAVEMENT_CONF_THRESH, verbose=False)[0]
S = np.zeros((h, w), dtype=np.float32)

GAUSS_ALPHA = 0.35

if seg_results.boxes is not None:
    boxes = seg_results.boxes.xyxy.cpu().numpy()
    scores = seg_results.boxes.conf.cpu().numpy()
    classes = seg_results.boxes.cls.cpu().numpy()

    for box, score, cls_id in zip(boxes, scores, classes):
        if int(cls_id) == PAVEMENT_CLASS_ID and score >= PAVEMENT_CONF_THRESH:
            x1, y1, x2, y2 = map(int, box)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            sigma_x = GAUSS_ALPHA * bw
            sigma_y = GAUSS_ALPHA * bh

            ys = np.arange(y1, y2)
            xs = np.arange(x1, x2)
            xv, yv = np.meshgrid(xs, ys)

            gauss = np.exp(
                -(((xv - cx) ** 2) / (2 * sigma_x ** 2)
                  + ((yv - cy) ** 2) / (2 * sigma_y ** 2))
            )

            S[y1:y2, x1:x2] = np.maximum(
                S[y1:y2, x1:x2],
                score * gauss
            )

S = np.clip(S, 0.0, 1.0)
np.save(os.path.join(SEG_DIR, fname.replace(".png", ".npy")), S)

print("[DONE] Single image processed successfully.")