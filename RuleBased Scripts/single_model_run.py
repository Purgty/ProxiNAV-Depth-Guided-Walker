from ultralytics import YOLO
import cv2
import os

model = YOLO(r"Pavement_Dataset_Model\train2\weights\best.pt")

video_path = r"Assets\pavement_visible_clip.mp4"
out_dir = r"Output\annotated_frames_pavement_val"
os.makedirs(out_dir, exist_ok=True)

# Stream frames
results = model(video_path, stream=True)

for i, result in enumerate(results):
    # Get annotated frame
    frame = result.plot()

    # Save each frame as image
    cv2.imwrite(os.path.join(out_dir, f"frame_{i:06d}.jpg"), frame)
