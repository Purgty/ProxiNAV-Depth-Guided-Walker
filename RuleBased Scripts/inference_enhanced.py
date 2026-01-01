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
    return (x_min + x_max) / 2, (y_min + y_max) / 2


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
    img_h, img_w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Depth Estimation ---
    input_image = Image.fromarray(frame_rgb)
    input_tensor = midas_transform(input_image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = midas_model(input_tensor)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=(img_h, img_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Normalize depth for visualization
    min_val, max_val = np.percentile(prediction, (2, 98))
    formatted_depth = ((prediction - min_val) / (max_val - min_val + 1e-6) * 255).astype("uint8")
    frame = cv2.addWeighted(frame, 0.7, cv2.cvtColor(formatted_depth, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # --- YOLO Inference ---
    obstacle_results = obstacle_model(frame, verbose=False)
    pavement_results = pavement_model(frame, verbose=False)
    curb_results = curb_segmentation_model(frame, verbose=False)

    all_detections = []

    # Obstacles
    obstacle_offset = 0
    for r in obstacle_results:
        for box in r.boxes:
            conf = box.conf.item()
            if conf > conf_threshold:
                cls_id = int(box.cls.item()) + obstacle_offset
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_detections.append([x1, y1, x2, y2, conf, cls_id])

    # Pavement
    pavement_offset = len(obstacle_model.names)
    for r in pavement_results:
        for box in r.boxes:
            conf = box.conf.item()
            if conf > conf_threshold:
                cls_id = int(box.cls.item()) + pavement_offset
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                all_detections.append([x1, y1, x2, y2, conf, cls_id])

    # Curbs
    curb_offset = pavement_offset + len(pavement_model.names)
    for r in curb_results:
        if r.masks is not None:
            for i, mask_tensor in enumerate(r.masks.data):
                conf = r.boxes[i].conf.item()
                if conf > conf_threshold:
                    cls_id = int(r.boxes[i].cls.item()) + curb_offset
                    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                    resized_mask = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                    coords = np.where(resized_mask > 0)
                    if coords[0].size > 0:
                        y_min, y_max = np.min(coords[0]), np.max(coords[0])
                        x_min, x_max = np.min(coords[1]), np.max(coords[1])
                        color = (255, 165, 0)
                        mask_overlay = np.zeros_like(frame)
                        mask_overlay[resized_mask > 0] = color
                        frame = cv2.addWeighted(frame, 1, mask_overlay, 0.4, 0)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 3)

    # --- NMS ---
    filtered_obstacles = []
    if all_detections:
        detections_tensor = torch.tensor(all_detections, dtype=torch.float32).cpu()
        nms = non_max_suppression(
            prediction=detections_tensor.unsqueeze(0),
            conf_thres=conf_threshold,
            iou_thres=iou_threshold
        )
        if nms[0] is not None and len(nms[0]) > 0:
            for det in nms[0].cpu().numpy():
                x1, y1, x2, y2, conf, cls_id = det.tolist()
                name = unified_class_names[int(cls_id)]
                bbox = [x1, y1, x2, y2]
                filtered_obstacles.append((bbox, conf, name))

                text = f"{name} C:{conf:.2f}"
                text_y = max(int(y1) - 10, 40)  # prevent off-screen text

                # Draw black shadow for bold effect
                cv2.putText(frame, text, (int(x1)+3, text_y+3),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)
                # Main colored text
                cv2.putText(frame, text, (int(x1), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

                # Bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 3)

    # --- Safe Zone ---
    center_x = img_w / 2
    zone_half_w = img_w * sensitivity_width_ratio / 2
    min_x, max_x = center_x - zone_half_w, center_x + zone_half_w
    cv2.line(frame, (int(min_x), 0), (int(min_x), img_h), (255, 255, 0), 2)
    cv2.line(frame, (int(max_x), 0), (int(max_x), img_h), (255, 255, 0), 2)
    cv2.putText(frame, "Fixed Safe Zone", (int(center_x) - 100, img_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

    # --- Guidance Logic ---
    message = "Path clear, continue straight."
    for bbox, conf, name in filtered_obstacles:
        x1, y1, x2, y2 = bbox
        c_x, _ = get_bbox_center(bbox)
        depth_roi = prediction[int(y1):int(y2), int(x1):int(x2)]
        depth = np.median(depth_roi) if depth_roi.size > 0 else 0

        if min_x < c_x < max_x:
            if depth < 1.5:
                urgency = "SHARP"
            elif depth < 3.0:
                urgency = "QUICK"
            elif depth < 5.0:
                urgency = "SLIGHT"
            else:
                continue  # far away, ignore

            if c_x < center_x:
                message = f"{urgency} right: {name} ahead."
            else:
                message = f"{urgency} left: {name} ahead."
            break

    # --- Bold guidance message with shadow ---
    cv2.putText(frame, message, (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 6)  # shadow
    cv2.putText(frame, message, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)  # main

    return frame, message


# --- Main Video Processing ---
if __name__ == "__main__":
    # Load models
    obstacle_model = YOLO(r'C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\WOTR_Dataset_Model\best.pt')
    pavement_model = YOLO(r'C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\Pavement_Dataset_Model\train1\weights\best.pt')
    curb_model = YOLO(r'C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\Pavement_Dataset_Model\best.pt')
    print("✅ YOLO Models loaded successfully!")

    print("Loading MiDaS model...")
    midas_model = torch.hub.load('intel-isl/MiDaS', 'DPT_Hybrid', pretrained=True)
    midas_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas_model.to(device)
    midas_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("✅ MiDaS loaded!")

    unified_class_names = list(obstacle_model.names.values()) + \
                          list(pavement_model.names.values()) + \
                          list(curb_model.names.values())

    # --- Video List ---
    video_list = [
        r"C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\Assets\mist_bridge.mp4",
        r"C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\Assets\rain_lowlight.mp4",
        r"C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\Assets\snowy.mp4",
        r"C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\Assets\tnagar_ppl.mp4",
        r"C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\Assets\widepavement_farobs.mp4"
    ]

    # --- Output Directory ---
    base_output = r"C:\Users\aswin\OneDrive\Desktop\Sandbox\AudioPavementWalkerPaper\processed_frames"
    os.makedirs(base_output, exist_ok=True)

    # --- Process Each Video ---
    for video_path in video_list:
        if not os.path.exists(video_path):
            print(f"⚠️ Skipping missing file: {video_path}")
            continue

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(base_output, video_name)
        version = 1
        while os.path.exists(output_dir):
            output_dir = os.path.join(base_output, f"{video_name}_v{version}")
            version += 1
        os.makedirs(output_dir)

        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        print(f"\n▶️ Processing video: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame, guidance = provide_guidance(
                frame,
                obstacle_model,
                pavement_model,
                curb_model,
                midas_model,
                midas_transform,
                device,
                unified_class_names
            )

            frame_out = os.path.join(output_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(frame_out, processed_frame)
            print(f"Saved {frame_out} | {guidance}")

            # Optionally display live video
            cv2.imshow("Audio Guided Walking System", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_id += 1

        cap.release()
        print(f"✅ Completed {video_name}, frames saved in {output_dir}")

    cv2.destroyAllWindows()