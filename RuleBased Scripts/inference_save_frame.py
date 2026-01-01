import cv2
import numpy as np
import torch
import math
import os
from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
from ultralytics.utils.ops import non_max_suppression

# --- Helper Functions ---
def get_bbox_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return center_x, center_y

def calculate_distance_euclidean(point1, point2):
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


# --- Main Guidance Logic ---
def provide_guidance(
    frame,
    obstacle_model,
    pavement_model,
    curb_segmentation_model,
    midas_model,
    midas_transform,
    device,
    unified_class_names,
    sensitivity_width_ratio=0.5,
    iou_threshold=0.45,
    conf_threshold=0.25
):
    """
    Runs inference on a frame, detects obstacles, pavement, walls, and curbs,
    estimates depth using MiDaS, applies cross-model NMS, and provides guidance.
    """
    img_height, img_width, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- 1. Depth Estimation using MiDaS ---
    input_image = Image.fromarray(frame_rgb)
    input_tensor = midas_transform(input_image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = midas_model(input_tensor)

    # Resize prediction to match original frame
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(img_height, img_width),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Normalize depth
    min_val = np.percentile(prediction, 2)
    max_val = np.percentile(prediction, 98)
    formatted_depth = ((prediction - min_val) / (max_val - min_val + 1e-6) * 255).astype("uint8")

    depth_display = cv2.cvtColor(formatted_depth, cv2.COLOR_GRAY2BGR)
    alpha = 0.3
    frame = cv2.addWeighted(frame, 1 - alpha, depth_display, alpha, 0)

    # --- 2. Run YOLO Models ---
    obstacle_results = obstacle_model(frame, verbose=False)
    pavement_results = pavement_model(frame, verbose=False)
    curb_segmentation_results = curb_segmentation_model(frame, verbose=False)

    all_detections = []

    # Obstacles
    obstacle_class_offset = 0
    for r in obstacle_results:
        for box in r.boxes:
            conf = box.conf.item()
            if conf > conf_threshold:
                cls_id_original = int(box.cls.item())
                unified_cls_id = cls_id_original + obstacle_class_offset
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_detections.append([x1, y1, x2, y2, conf, unified_cls_id])

    # Pavement / Road / Wall
    pavement_class_offset = len(obstacle_model.names)
    for r in pavement_results:
        for box in r.boxes:
            conf = box.conf.item()
            if conf > conf_threshold:
                cls_id_original = int(box.cls.item())
                unified_cls_id = cls_id_original + pavement_class_offset
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_detections.append([x1, y1, x2, y2, conf, unified_cls_id])

    # Curb Segmentation
    filtered_curb_segments = []
    curb_class_offset = pavement_class_offset + len(pavement_model.names)

    for r in curb_segmentation_results:
        if r.masks is not None:
            for i, mask_tensor in enumerate(r.masks.data):
                conf = r.boxes[i].conf.item()
                if conf > conf_threshold:
                    cls_id_original = int(r.boxes[i].cls.item())
                    curb_name = curb_segmentation_model.names[cls_id_original]
                    unified_cls_id = cls_id_original + curb_class_offset

                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    resized_mask = cv2.resize(mask_np, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

                    coords = np.where(resized_mask > 0)
                    if coords[0].size > 0:
                        y_min, y_max = np.min(coords[0]), np.max(coords[0])
                        x_min, x_max = np.min(coords[1]), np.max(coords[1])
                        mask_bbox = [x_min, y_min, x_max, y_max]
                        depth_roi = prediction[y_min:y_max+1, x_min:x_max+1]
                        curb_depth_value = np.median(depth_roi) if depth_roi.size > 0 else 0

                        filtered_curb_segments.append({
                            'mask': resized_mask,
                            'name': curb_name,
                            'depth': curb_depth_value,
                            'bbox': mask_bbox,
                            'unified_cls_id': unified_cls_id
                        })

                        color = (255, 165, 0)
                        mask_overlay = np.zeros_like(frame, dtype=np.uint8)
                        mask_overlay[resized_mask > 0] = color
                        frame = cv2.addWeighted(frame, 1, mask_overlay, 0.4, 0)

                        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
                        cv2.putText(frame, f"{curb_name} C:{conf:.2f} D:{curb_depth_value:.2f}",
                                    (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --- NMS ---
    filtered_pavement_boxes = []
    filtered_obstacle_boxes = []
    filtered_wall_boxes = []

    if all_detections:
        detections_tensor = torch.tensor(all_detections, dtype=torch.float32).cpu()
        nms_results = non_max_suppression(
            prediction=detections_tensor.unsqueeze(0),
            conf_thres=conf_threshold,
            iou_thres=iou_threshold
        )
        filtered_detections_bbox = nms_results[0].cpu().numpy()

        wall_original_names = ['wall']
        wall_unified_ids = [
            idx + pavement_class_offset for idx, name in enumerate(pavement_model.names.values())
            if name in wall_original_names
        ]

        for det in filtered_detections_bbox:
            x1, y1, x2, y2, conf, unified_cls_id = det.tolist()
            class_name = unified_class_names[int(unified_cls_id)]
            bbox = [x1, y1, x2, y2]

            if unified_cls_id < pavement_class_offset:
                filtered_obstacle_boxes.append(bbox + [conf, class_name])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_name} C:{conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            elif unified_cls_id in wall_unified_ids:
                filtered_wall_boxes.append(bbox + [conf, class_name])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name} C:{conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            else:
                filtered_pavement_boxes.append(bbox + [conf, class_name])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} C:{conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # --- Fixed Safe Zone ---
    safe_zone_half_width = img_width * sensitivity_width_ratio / 2
    frame_center_x = img_width / 2
    safe_zone_min_x = frame_center_x - safe_zone_half_width
    safe_zone_max_x = frame_center_x + safe_zone_half_width

    cv2.line(frame, (int(safe_zone_min_x), 0), (int(safe_zone_min_x), img_height), (255, 255, 0), 2)
    cv2.line(frame, (int(safe_zone_max_x), 0), (int(safe_zone_max_x), img_height), (255, 255, 0), 2)
    cv2.putText(frame, "Fixed Safe Zone", (int(frame_center_x) - 60, img_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # --- Corrected Guidance Logic ---
    guidance_message = ""

    obstacles_with_depth = []
    for obs_bbox in filtered_obstacle_boxes:
        x1, y1, x2, y2 = obs_bbox[:4]
        depth_roi = prediction[int(y1):int(y2), int(x1):int(x2)]
        obs_depth_value = np.median(depth_roi) if depth_roi.size > 0 else 0
        obs_center_x, obs_center_y = get_bbox_center(obs_bbox[:4])
        obstacles_with_depth.append({
            'name': obs_bbox[5],
            'center_x': obs_center_x,
            'center_y': obs_center_y,
            'depth': obs_depth_value,
            'bbox': obs_bbox[:4]
        })

    obstacles_with_depth.sort(key=lambda x: x['depth'])  # smaller = closer

    obstacle_in_safe_zone_found = False
    for obs in obstacles_with_depth:
        if safe_zone_min_x < obs['center_x'] < safe_zone_max_x:
            if obs['depth'] >= 5.0:
                continue  # ignore far obstacles

            obstacle_in_safe_zone_found = True
            if obs['depth'] < 1.5:
                urgency = "SHARP"
            elif obs['depth'] < 3.0:
                urgency = "QUICK"
            else:
                urgency = "SLIGHT"

            if obs['center_x'] < frame_center_x:
                guidance_message = f"{urgency} right: {obs['name']} ahead."
            else:
                guidance_message = f"{urgency} left: {obs['name']} ahead."
            break

    if not obstacle_in_safe_zone_found:
        guidance_message = "Path clear, continue straight."

    # Display and print message
    cv2.putText(frame, guidance_message, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    print(f"[GUIDANCE] {guidance_message}")

    return frame


# --- Main Execution ---
if __name__ == "__main__":
    try:
        obstacle_model = YOLO(r'WOTR_Dataset_Model\best.pt')
        pavement_model = YOLO(r'Pavement_Dataset_Model\train1\weights\best.pt')
        # curb_segmentation_model = YOLO(r'Pavement_Dataset_Model\best.pt')
        print("YOLO Models loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO models: {e}")
        exit()

    # --- Load MiDaS Depth Model ---
    try:
        print("Loading MiDaS DPT_Hybrid depth model via torch.hub...")
        midas_model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', pretrained=True)
        midas_model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        midas_model.to(device)

        midas_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        print("✅ MiDaS depth model loaded successfully!")
    except Exception as e:
        print(f"Error loading MiDaS model: {e}")
        exit()

    # --- Unified Class Names ---
    obstacle_names = list(obstacle_model.names.values())
    pavement_wall_names = list(pavement_model.names.values())
    # curb_names = list(curb_segmentation_model.names.values())
    unified_class_names = obstacle_names + pavement_wall_names    # + curb_names

    # --- Image Paths ---
    input_image_path = r'Assets\pavement2.png'
    output_image_path = r'Assets\pavement2_output.jpg'

    img = cv2.imread(input_image_path)
    if img is None:
        print(f"Error: Could not open image at {input_image_path}")
        exit()

    processed_img = provide_guidance(
        img.copy(),
        obstacle_model,
        pavement_model,
        # curb_segmentation_model,
        midas_model,
        midas_transform,
        device,
        unified_class_names
    )
    success = cv2.imwrite(output_image_path, processed_img)
    if not success:
        print(f"Error: Failed to save image to {output_image_path}")
    else:
        print(f"Output image saved to {output_image_path}")

    window_name = 'Audio Guided Walking System - Single Image'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)  # ensures native image size
    cv2.imshow(window_name, processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()