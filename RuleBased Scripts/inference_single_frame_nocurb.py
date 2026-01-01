import cv2
import numpy as np
import torch
import math
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from ultralytics.utils.ops import non_max_suppression


# --- Helper Functions ---
def get_bbox_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    return (x_min + x_max) / 2, (y_min + y_max) / 2


# --- Main Guidance Logic ---
def provide_guidance(
    frame,
    obstacle_model,
    pavement_model,
    midas_model,
    midas_transform,
    device,
    unified_class_names,
    sensitivity_width_ratio=0.5,
    iou_threshold=0.45,
    conf_threshold=0.25
):
    img_height, img_width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- 1. Depth Estimation (MiDaS) ---
    input_tensor = midas_transform(Image.fromarray(frame_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = midas_model(input_tensor)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(img_height, img_width),
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()

    min_val, max_val = np.percentile(prediction, (2, 98))
    depth_norm = ((prediction - min_val) / (max_val - min_val + 1e-6) * 255).astype(np.uint8)
    depth_color = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

    frame = cv2.addWeighted(frame, 0.7, depth_color, 0.3, 0)

    # --- 2. YOLO Detection ---
    obstacle_results = obstacle_model(frame, verbose=False)
    pavement_results = pavement_model(frame, verbose=False)

    all_detections = []

    # Obstacles
    for r in obstacle_results:
        for box in r.boxes:
            if box.conf.item() > conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_detections.append([
                    x1, y1, x2, y2,
                    box.conf.item(),
                    int(box.cls.item())
                ])

    # Pavement / Wall
    offset = len(obstacle_model.names)
    for r in pavement_results:
        for box in r.boxes:
            if box.conf.item() > conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_detections.append([
                    x1, y1, x2, y2,
                    box.conf.item(),
                    int(box.cls.item()) + offset
                ])

    # --- NMS ---
    if all_detections:
        det_tensor = torch.tensor(all_detections, dtype=torch.float32).unsqueeze(0)
        nms_out = non_max_suppression(det_tensor, conf_threshold, iou_threshold)[0].cpu().numpy()

        for x1, y1, x2, y2, conf, cls_id in nms_out:
            cls_id = int(cls_id)
            label = unified_class_names[cls_id]

            color = (0, 0, 255) if cls_id < offset else (0, 255, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- Safe Zone ---
    half_width = img_width * sensitivity_width_ratio / 2
    cx = img_width / 2
    cv2.line(frame, (int(cx - half_width), 0), (int(cx - half_width), img_height), (255, 255, 0), 2)
    cv2.line(frame, (int(cx + half_width), 0), (int(cx + half_width), img_height), (255, 255, 0), 2)

    # --- Guidance ---
    guidance = "Path clear, continue straight."
    cv2.putText(frame, guidance, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame


# --- Main Execution ---
if __name__ == "__main__":
    obstacle_model = YOLO(r'WOTR_Dataset_Model\best.pt')
    pavement_model = YOLO(r'Pavement_Dataset_Model\train1\weights\best.pt')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    midas_model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", pretrained=True)
    midas_model.to(device).eval()

    midas_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    unified_class_names = (
        list(obstacle_model.names.values()) +
        list(pavement_model.names.values())
    )

    input_image_path = r'Assets\suburbs3.png'
    output_image_path = r'Assets\suburbs3_output.jpg'

    img = cv2.imread(input_image_path)
    if img is None:
        raise RuntimeError("Image not found")

    output = provide_guidance(
        img.copy(),
        obstacle_model,
        pavement_model,
        midas_model,
        midas_transform,
        device,
        unified_class_names
    )

    cv2.imwrite(output_image_path, output)
    cv2.imshow("Guidance Output", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()